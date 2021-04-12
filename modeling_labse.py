import torch

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutputWithPoolingAndCrossAttentions
from transformers import EncoderDecoderModel, RobertaModel, XLMRobertaModel, XLMRobertaTokenizer
from torch.nn.functional import normalize

from train_utils import AMSLoss, compute_loss


class LARO(XLMRobertaModel):

    def prepare_freezed_forward(self):
        for m in self.named_parameters():
            if "layer" in m[0].lower() and "embeddings" not in m[0].lower():
                num_layers = m[0].lower().split(".")[3]
        for m in self.named_parameters():
            if "embeddings" in m[0].lower():
                m[1].requires_grad = False
            if "norm" in m[0].lower() or "layer.0" in m[0].lower() or "layer." + str(
                    num_layers) in m[0].lower():
                m[1].requires_grad = True
            else:
                m[1].requires_grad = False

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).decoder_input_ids=None,
                                  decoder_attention_mask=None,
        """
        self.prepare_freezed_forward()

        scaling_factor = 10

        output_1 = torch.mul(scaling_factor,normalize(super(LARO, self).forward(input_ids=input_ids,
                                             attention_mask=attention_mask).last_hidden_state[:, 0, :]))

        output_2 = torch.mul(scaling_factor,normalize(super(LARO, self).forward(input_ids=decoder_input_ids,
                                             attention_mask=decoder_attention_mask).last_hidden_state[:, 0, :]))

        loss = compute_loss(output_1, output_2)

        print(loss.item())

        return (loss,)


    @torch.no_grad()
    def get_embedding(self, input_ids=None, attention_mask=None) -> torch.Tensor:
        """
        generates a embedding from the encoder
        :param input_ids: tokenized sentences
        :param attention_mask: masking of tokens
        :return: Tensor -> Embeddings
        """
        encoder_outputs = super(LARO, self).forward(input_ids=input_ids,
                                                    attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return normalize(encoder_outputs)


if __name__ == "__main__":
    from transformers import BertTokenizer, EncoderDecoderModel

    model = LARO.from_pretrained('xlm-roberta-large')
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

    input_ids = tokenizer(["Hello, my dog is cute", "Hello, my dog is cute"], add_special_tokens=True, truncation=True,
                          return_tensors="pt", padding=True)
    outputs = model(input_ids=input_ids["input_ids"], decoder_input_ids=input_ids["input_ids"],
                    attention_mask=input_ids["attention_mask"], decoder_attention_mask=input_ids["attention_mask"])
    # training

    # save and load from pretrained
    # model.save_pretrained("bert2bert")
    # model = EncoderDecoderModel.from_pretrained("bert2bert")
    # generation
