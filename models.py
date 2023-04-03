# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 20:05:24 2023

@author: Admin
"""
import torch
import torch.nn as nn


class TranslationModel(nn.Module):
    def __init__(
        self,
        input_l,
        output_l,
        n_token,
        encoder_layer=6,
        decoder_layer=6,
        d=512,
        n_head=8,
        sos_id=1,
        pad_id=0,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_l, n_token, n_layer=encoder_layer, d=d, n_head=n_head, pad_id=pad_id
        )
        self.decoder = Decoder(
            output_l,
            input_l,
            n_token,
            n_layer=decoder_layer,
            d=d,
            n_head=n_head,
            sos_id=sos_id,
            pad_id=pad_id,
        )

    def forward(self, inputs, outputs=None, beam=1):
        feature = self.encoder(inputs)  # [B,S,512]
        if outputs is None:
            return self.decoder(feature, caption=None, top_k=beam)
        return self.decoder(feature, outputs)  # [B,L,n_token]


class Encoder(nn.Module):
    def __init__(self, max_l, n_token, n_layer=6, d=512, n_head=8, pad_id=0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_head)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.posit_embedding = nn.Embedding(max_l, d)
        self.token_embedding = nn.Embedding(n_token, d)
        self.nhead = n_head
        self.pad_id = pad_id

    def forward(self, inputs):
        posit_index = (
            torch.arange(inputs.shape[1]).unsqueeze(0).repeat(inputs.shape[0], 1).to(inputs.device)
        )  # (B,S)
        source_posit_embed = self.posit_embedding(posit_index)
        padding_mask = inputs == self.pad_id

        inputs = self.token_embedding(inputs)  # [B,S,512]

        source_embed = inputs + source_posit_embed
        source_embed = torch.transpose(source_embed, 0, 1)
        attn_mask = torch.full((inputs.shape[1], inputs.shape[1]), 0.0).to(inputs.device)

        output = self.transformer_encoder(
            src=source_embed, mask=attn_mask, src_key_padding_mask=padding_mask
        )  # [S, B, 512]
        output = torch.transpose(output, -2, -3)  # [B, S, 512]
        return output


class Decoder(nn.Module):
    def __init__(self, max_l, input_l, n_token, sos_id=1, pad_id=0, n_layer=6, n_head=8, d=512):
        super().__init__()
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.n_head = n_head
        self.d = d
        if n_token is not None:
            self.n_token = n_token
            self.token_embedding = nn.Embedding(n_token, d)
        self.posit_embedding = nn.Embedding(max_l, d)
        self.source_posit_embedding = nn.Embedding(input_l, d)

        self.max_l = max_l

        decoder_layer = nn.TransformerDecoderLayer(d_model=d, nhead=n_head)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)

        if n_token is not None:
            self.output = nn.Linear(d, n_token)
            self.output.weight = self.token_embedding.weight

    def forward(self, source, caption, top_k=1, eos_id=2, mode='greedy'):
        """
        source: [B,S,E], S=1 or n_slice
        caption: [B,L], token index。
        """
        if caption is None:
            return self._infer(source=source, top_k=top_k, eos_id=eos_id, mode=mode)  # (B,l)

        posit_index = (
            torch.arange(caption.shape[1])
            .unsqueeze(0)
            .repeat(caption.shape[0], 1)
            .to(caption.device)
        )  # (B,L)
        target_embed = self.posit_embedding(posit_index)  # 输入shape后面增加E。(B,L,E)
        target_embed += self.token_embedding(caption)  # (B,L,E)
        padding_mask = caption == self.pad_id  # [B,L]

        attn_mask = self.generate_square_subsequent_mask(caption.shape[1]).to(
            caption.device
        )  # [L,L]

        # posit_index = torch.arange(source.shape[1]).unsqueeze(0).repeat(caption.shape[0],1).to(source.device) #(B,S)
        # source_posit_embed = self.source_posit_embedding(posit_index) # [B,S,E]
        # source_embed = source + source_posit_embed

        target_embed = torch.transpose(target_embed, 0, 1)
        source_embed = torch.transpose(source, 0, 1)
        out = self.transformer_decoder(
            tgt=target_embed,
            memory=source_embed,
            tgt_mask=attn_mask,
            tgt_key_padding_mask=padding_mask,
        )

        out = torch.transpose(out, -2, -3)  # [B, L, E]
        out = self.output(out)  # [B, L, n_token]
        return out

    def _infer(self, source, top_k=1, eos_id=2, mode='greedy'):
        """
        source: [B,S,E],
        """
        outputs = (
            torch.ones((source.shape[0], 1), dtype=torch.long).to(source.device) * self.sos_id
        )  # (K,B,1) SOS
        not_over = torch.ones((source.shape[0])).to(source.device)  # [K,B]
        assert top_k == 1

        for _ in range(1, self.max_l):
            out = self.forward(source, outputs)  # [B, L, n_token]
            prob = nn.functional.softmax(out, dim=2)[:, -1]  # [B, n_token]
            val, idx = torch.topk(prob, 1)  # (B,1)

            outputs = torch.cat([outputs, idx[:, 0].view(-1, 1)], dim=-1)  # (B,L+1)
            not_over = torch.minimum(not_over, torch.ne(outputs[:, -1], eos_id).long())  # [B]
            if torch.sum(not_over) == 0:
                break
        return outputs  # (B,L)

    def generate_square_subsequent_mask(self, sz):
        # float mask, -inf无法关注，0可以关注
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)  # 1。预测i位置可以用i及之前位置的输入
