# Copyright 2022   Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
"""
MAE network.
"""
from mindspore import nn
import mindspore.numpy as np
from mindspore import Tensor
from mindspore import ops as P
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.nn.transformer import TransformerDecoder, TransformerEncoder
from mindspore.common.initializer import initializer, XavierUniform

from utils.helper import get_2d_sin_cos_pos_embed
from model.modules import Patchify, PatchEmbed


class MAEModule(nn.Cell):
    """
    MAE network definition.

    Args:
        batch_size (int): Batch size.
        image_size (int): Image size.
        patch_size (int): Patch size.
        masking_ratio (float): Percentage of mask. Default: 0.75.
        channels (int): Number of channel. Default: 3.
    """

    def __init__(self,
                 batch_size,
                 image_size,
                 patch_size,
                 masking_ratio=0.75,
                 channels=3):
        super(MAEModule, self).__init__()
        assert 0 < masking_ratio < 1, 'masking ratio must be between 0 and 1'

        # Number of patches in a picture
        self.num_patches = (image_size // patch_size) ** 2

        # Number of patches masked in a picture
        self.num_masked = int(self.num_patches * masking_ratio)

        # Raise the dimension, e.g. array([0, 1, 2]) becomes array([[0], [1], [2]])
        self.batch_range = np.arange(batch_size)[:, None]

        # The sum of the number of channels of all pixels in a patch
        self.patch_dim = channels * patch_size ** 2

        # Calculate the patches to be masked and obtain random indexes for classification as masked and unmasked
        self.rand_indices = P.Sort()(P.UniformReal()((batch_size, self.num_patches)))
        self.masked_indices = self.rand_indices[1][:, :self.num_masked]
        self.unmasked_indices = self.rand_indices[1][:, self.num_masked:]
        self.mask_info = None
        self.encoder = None
        self.decoder = None

    def generate_mask(self):
        """
        Get information about the mask, such as the range of the batch,
        the index of the mask and the index of the unmask.

        Returns:
            array, Dimension of batch.
            int, Number of mask indices.
            int, Number of unmasked indices.
        """
        self.mask_info = {
            "batch_range": self.batch_range,
            "mask_indices": self.masked_indices,
            "unmasked_indices": self.unmasked_indices
        }
        return self.mask_info

    def encoder_engine(self):
        """
        Get MAE model encoder information.

        Returns:
            network, MAE's encoder structure.
        """
        return self.encoder

    def decoder_engine(self):
        """
        Get MAE model encoder information.

        Returns:
            network, MAE's encoder structure.
        """
        return self.decoder


class MAEEncoder(MAEModule):
    """
    MAE Encoder network definition.

    Args:
        batch_size (int): Batch size.
        image_size (int): int, Image size.
        patch_size (int): int, Patch size.
        encoder_layers (int): int, Number of encoder layer. Default: 12.
        encoder_num_heads (int): int, Number of encoder head. Default: 12.
        encoder_dim (int): Dimension of encoder. Default: 768.
        mlp_ratio (int): int, Multiplicative between encoder dimension and fully connected layer. Default: 4.
        masking_ratio (float): float, Percentage of mask. Default: 0.75.
        drop_path (float): float, Percentage of drop path. Default: 0.1.
        channels (int): int, Number of channel. Default: 3.
        initialization (initializer): Parameter initialization method.

    Returns:
        Tensor, output tensor, the shape is (batch_size, encoder_dim).
    """

    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 mlp_ratio=4,
                 masking_ratio=0.75,
                 drop_path=0.1,
                 channels=3,
                 initialization=XavierUniform()
                 ):
        super(MAEEncoder, self).__init__(batch_size, image_size, patch_size, masking_ratio, channels)

        # The length of the input sequence in the encoder, the input sequence is the unmasked patch.
        self.seq_length = self.num_patches - self.num_masked + 1
        self.encoder = TransformerEncoder(batch_size=batch_size,
                                          num_layers=encoder_layers,
                                          num_heads=encoder_num_heads,
                                          hidden_size=encoder_dim,
                                          ffn_hidden_size=encoder_dim * mlp_ratio,
                                          seq_length=self.seq_length,
                                          hidden_dropout_rate=drop_path)

        # cls token
        cls_token = Parameter(
            initializer(initialization, (1, 1, encoder_dim)),
            name='cls', requires_grad=True
        )
        self.cls_token = P.Tile()(cls_token, (batch_size, 1, 1))

        # pos embedding
        self.encoder_pos_embedding = Parameter(
            initializer(initialization, (1, self.num_patches + 1, encoder_dim)),
            name='pos_embedding', requires_grad=False
        )
        self.add = P.Add()
        self.cat = P.Concat(axis=1)

        # Input Tensor for slice extraction based on step size and index.
        self.stride_slice = P.StridedSlice()
        self.norm = nn.LayerNorm((encoder_dim,), epsilon=1e-6).to_float(mstype.float32)
        # Embedding operations on patches.
        self.patch_embed = PatchEmbed(img_size=image_size,
                                      patch_size=patch_size,
                                      in_features=channels,
                                      out_features=encoder_dim)

        self.encoder_input_mask = Tensor(np.ones((batch_size, self.seq_length, self.seq_length)), mstype.float32)

        # Initialization weights
        self._init_weights()

    def _init_weights(self):
        """Initialization weights"""
        encoder_pos_emd = Tensor(

            # Get the 2-dimensional sin-cos position code.
            get_2d_sin_cos_pos_embed(self.encoder_pos_embedding.shape[-1],
                                     int(self.num_patches ** .5),
                                     cls_token=True),
            mstype.float32
        )
        self.encoder_pos_embedding.set_data(

            # Add a dimension to the encoder's position encoding.
            P.ExpandDims()(encoder_pos_emd, 0)
        )

    def construct(self, img):
        """ build network """

        # Change patch to token and add location code.
        tokens = self.patch_embed(img)
        encoder_pos_embedding = self.stride_slice(self.encoder_pos_embedding, (0, 1, 0),
                                                  (1, self.encoder_pos_embedding.shape[1],
                                                   self.encoder_pos_embedding.shape[2]), (1, 1, 1))
        tokens = self.add(tokens, encoder_pos_embedding)

        # Get the unmasked token.
        tokens = tokens[self.batch_range, self.unmasked_indices]

        # Get the cls_token and add the location code.
        cls_pos_embedding = self.stride_slice(self.encoder_pos_embedding, (0, 0, 0),
                                              (1, 1, self.encoder_pos_embedding.shape[2]),
                                              (1, 1, 1))
        cls_tokens = self.add(self.cls_token, cls_pos_embedding)

        # Splice cls_token and tokens.
        tokens = self.cat((cls_tokens, tokens))

        # Get the final tokens to be encoded.
        encoded_tokens = self.encoder(tokens, self.encoder_input_mask)[0]
        encoded_tokens = self.norm(encoded_tokens)

        return encoded_tokens


class PreTrainMAEVit(MAEModule):
    """
    Establishment of pre training process of MAE model

    Args:
        batch_size (int): Batch size.
        patch_size (int): Patch size.
        image_size (int): Image size.
        encoder_layers (int): Number of encoder layer. Default: 12.
        decoder_layers (int): Number of decoder layer. Default: 8.
        encoder_num_heads (int): Number of encoder head. Default: 12.
        decoder_num_heads (int): Number of decoder head. Default: 16.
        encoder_dim (int): Dimension of encoder. Default: 768.
        decoder_dim (int): Dimension of decoder. Default: 512.
        mlp_ratio (int): Multiplicative between encoder dimension and fully connected layer. Default: 4.
        masking_ratio (float): Percentage of mask. Default: 0.75.
        drop_path (float): Percentage of drop path. Default: 0.1.
        channels (int): Number of channel. Default: 3.
        norm_pixel_loss (boolean): Whether to use the norm pixel loss. Default: False.
        initialization (initializer): Parameter initialization method.

    Returns:
        Tensor, output tensor, the shape is (batch_size, encoder_dim).
    """

    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 decoder_layers=8,
                 encoder_num_heads=12,
                 decoder_num_heads=16,
                 encoder_dim=768,
                 decoder_dim=512,
                 mlp_ratio=4,
                 masking_ratio=0.75,
                 drop_path=0.1,
                 channels=3,
                 norm_pixel_loss=False,
                 initialization=XavierUniform()):
        super(PreTrainMAEVit, self).__init__(batch_size, image_size, patch_size, masking_ratio, channels)
        self.encoder = MAEEncoder(batch_size, patch_size, image_size,
                                  encoder_layers=encoder_layers, encoder_dim=encoder_dim,
                                  encoder_num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
                                  drop_path=drop_path, initialization=initialization)

        # Parameters of the encoder.
        self.seq_length = self.encoder.seq_length
        tgt_seq_length = self.num_patches + 1
        self.mask_token = Parameter(P.StandardNormal()((decoder_dim,)))
        self.mask_tokens = P.Tile()(self.mask_token, (batch_size, self.num_masked, 1))

        # Fully connected layer from encoder to decoder.
        self.enc_to_dec = nn.Dense(encoder_dim, decoder_dim,
                                   has_bias=True) if encoder_dim != decoder_dim else P.Identity()
        self.decoder = TransformerDecoder(batch_size=batch_size, num_layers=decoder_layers,
                                          num_heads=decoder_num_heads, hidden_size=decoder_dim,
                                          ffn_hidden_size=decoder_dim * mlp_ratio,
                                          src_seq_length=self.seq_length,
                                          tgt_seq_length=tgt_seq_length)

        # Get the embedding of the decoder.
        decoder_pos_emd = Tensor(
            get_2d_sin_cos_pos_embed(decoder_dim, int(self.num_patches ** .5), cls_token=True), mstype.float32
        )

        # embedding of the decoder.
        self.decoder_pos_embedding = nn.Embedding(tgt_seq_length, decoder_dim,
                                                  embedding_table=decoder_pos_emd)
        self.decoder_pos_embedding.requires_grad = False
        self.attention_mask = Tensor(np.ones((batch_size, tgt_seq_length, tgt_seq_length)), mstype.float32)

        # Fully connected layer between decoder and patch.
        self.to_pixels = nn.Dense(decoder_dim, self.patch_dim, has_bias=True)
        self.decoder_norm = nn.LayerNorm((decoder_dim,), epsilon=1e-6).to_float(mstype.float32)
        self.patchify = Patchify(patch_size=patch_size)
        self.add = P.Add()
        self.divide = P.Div()
        self.cast = P.Cast()
        self.cat = P.Concat(axis=1)
        self.pow = P.Pow()
        self.mean = P.ReduceMean(keep_dims=True)
        self.norm_pixel_loss = norm_pixel_loss

        # loss specified as MSELoss.
        self.mse_loss = nn.MSELoss()

    def calc_loss(self, pred, target):
        """
        Calculated loss.

        Args:
            pred (int): Prediction result category.
            target (int): Real Results Category.

        Returns:
            Float, Number of MSELoss.
        """
        pred = self.cast(pred, mstype.float32)
        target = self.cast(target, mstype.float32)
        if self.norm_pixel_loss:
            mean = self.mean(target, -1)
            var = target.var(axis=-1, keepdims=True)
            target = self.divide(target - mean, self.pow(var + 1e-6, 0.5))
        recon_loss = self.mse_loss(pred, target)
        return recon_loss

    def construct(self, img):
        """ build network """

        # Get the patch to enter into the encoder.
        encoder_tokens = self.encoder(img)
        patches = self.patchify(img)

        # From encoder to decoder
        decoder_tokens = self.enc_to_dec(encoder_tokens)

        # Add a location code to the token of the encoder.
        img_tokens = decoder_tokens[:, 1:, :]
        cls_tokens = decoder_tokens[:, :1, :]
        decoder_tokens_ = self.add(img_tokens, self.decoder_pos_embedding(self.unmasked_indices))
        decoder_tokens = self.cat((cls_tokens, decoder_tokens_))

        # Add a location code to the masked token based on the index of the mask obtained above.
        mask_tokens = self.add(self.mask_tokens, self.decoder_pos_embedding(self.masked_indices))

        # Splice the encoder's token with the masked token.
        decoder_tokens = self.cat((decoder_tokens, mask_tokens))
        decoded_tokens = self.decoder(decoder_tokens, self.attention_mask)[0]

        # Normalize the token of the encoder.
        decoded_tokens = self.decoder_norm(decoded_tokens)

        # Project all tokens onto pixels
        decoded_tokens = decoded_tokens[:, 1:, :]
        pred_pixel_values = self.to_pixels(decoded_tokens)

        # Sort patches by index
        masked_patches = patches[self.batch_range, self.masked_indices]
        unmasked_patches = patches[self.batch_range, self.unmasked_indices]
        sort_patches = self.cat((unmasked_patches, masked_patches))

        # Calculate the loss of reconstruction
        loss = self.calc_loss(pred_pixel_values, sort_patches)
        return loss
