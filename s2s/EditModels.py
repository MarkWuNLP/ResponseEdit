import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import s2s.modules
from s2s.Models import StackedGRU
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

try:
    import ipdb
except ImportError:
    pass


class EditEncoder(nn.Module):
    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.hidden_size = opt.enc_rnn_size

        super(EditEncoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths)
        """
        wordEmb = self.word_lut(input[0])
        # emb = pack(wordEmb, lengths)
        hidden_t = torch.sum(wordEmb, dim=0)
        return hidden_t



class MultiSourceEditModel(nn.Module):
    def __init__(self, encoder, editEncoder, decoder, decIniter):
        super( MultiSourceEditModel, self).__init__()
        self.encoder = encoder
        self.editEncoder = editEncoder
        self.decoder = decoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, input):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths),
        (wrap(srcInsBatch), lengths), (wrap(srcDelBatch), lengths),
        (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch))
        """
        # ipdb.set_trace()
        src = input[0]
        src_ins = input[1][0]
        src_ins_len = Variable(src_ins.data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        src_del = input[2][0]
        src_del_len = Variable(src_del.data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        src_Que = input[3][0]
        src_Que_len = Variable(src_Que.data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)

        tgt = input[4][0][:-1]  # exclude last target from inputs
        src_pad_mask = Variable(src[0].data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        enc_hidden, context = self.encoder(src)


        # enc_ins_hidden = self.editEncoder(src_ins)
        # enc_del_hidden = self.editEncoder(src_del)

        init_att = self.make_init_att(context)
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden

        g_out, dec_hidden, _attn, _attention_vector = self.decoder(tgt, enc_hidden, src_ins, src_del,
                                                                   context, src_pad_mask, init_att,src_ins_len,src_del_len,src_Que, src_Que_len)

        return g_out


class IDEditModel(nn.Module):
    def __init__(self, encoder, editEncoder, decoder, decIniter):
        super(IDEditModel, self).__init__()
        self.encoder = encoder
        self.editEncoder = editEncoder
        self.decoder = decoder
        self.decIniter = decIniter

    def make_init_att(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.encoder.hidden_size * self.encoder.num_directions)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, input):
        """
        input: (wrap(srcBatch), wrap(srcBioBatch), lengths), 
        (wrap(srcInsBatch), lengths), (wrap(srcDelBatch), lengths),
        (wrap(tgtBatch), wrap(copySwitchBatch), wrap(copyTgtBatch))
        """
        # ipdb.set_trace()
        src = input[0]
        src_ins = input[1][0]
        src_ins_len = Variable(src_ins.data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        src_del = input[2][0]
        src_del_len = Variable(src_del.data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        tgt = input[3][0][:-1]  # exclude last target from inputs
        src_pad_mask = Variable(src[0].data.eq(s2s.Constants.PAD).transpose(0, 1).float(), requires_grad=False,
                                volatile=False)
        enc_hidden, context = self.encoder(src)

        # enc_ins_hidden = self.editEncoder(src_ins)
        # enc_del_hidden = self.editEncoder(src_del)

        init_att = self.make_init_att(context)
        enc_hidden = self.decIniter(enc_hidden[1]).unsqueeze(0)  # [1] is the last backward hiden

        g_out, dec_hidden, _attn, _attention_vector = self.decoder(tgt, enc_hidden, src_ins, src_del,
                                                                   context, src_pad_mask, init_att,src_ins_len,src_del_len)

        return g_out


class EditAttDecoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size + opt.word_vec_size * 2

        super(EditAttDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.attn2 = s2s.modules.ConcatAttention(opt.word_vec_size, opt.enc_rnn_size,opt.word_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size
                                  + opt.dec_rnn_size
                                  + opt.word_vec_size
                                  + opt.word_vec_size * 2 # edit vector length
                                  ), opt.dec_rnn_size)
        self.maxout = s2s.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.hidden_size = opt.dec_rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, ins_hidden, del_hidden, context, src_pad_mask, init_att, ins_pad_mask, del_pad_mask):
        emb = self.word_lut(input)

        self.attn2.applyMask(ins_pad_mask)
        wordEmb = self.word_lut(ins_hidden)
        ins_hidden = self.attn2(context[-1],wordEmb.transpose(0, 1), None)[0]
        #ins_hidden = self.attn2(init_att,wordEmb.transpose(0, 1), None)[0]
        #ins_hidden = torch.sum(wordEmb, dim=0)
        self.attn2.applyMask(del_pad_mask)
        wordEmb = self.word_lut(del_hidden)
        #del_hidden = self.attn2(init_att,wordEmb.transpose(0, 1), None)[0]
        del_hidden = self.attn2(context[-1],wordEmb.transpose(0, 1), None)[0]
        #del_hidden = torch.sum(wordEmb, dim=0)




        g_outputs = []
        cur_context = init_att

        self.attn.applyMask(src_pad_mask)
        precompute = None
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context, ins_hidden, del_hidden], 1)
            output, hidden = self.rnn(input_emb, hidden)
            cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

            readout = self.readout(torch.cat((emb_t, output, cur_context, ins_hidden, del_hidden), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
        g_outputs = torch.stack(g_outputs)
        return g_outputs, hidden, attn, cur_context


class EditMultiSourceAttDecoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size + opt.word_vec_size * 3

        super(EditMultiSourceAttDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.attn2 = s2s.modules.ConcatAttention(opt.word_vec_size, opt.enc_rnn_size,opt.word_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size
                                  + opt.dec_rnn_size
                                  + opt.word_vec_size
                                  + opt.word_vec_size * 3 # edit vector length
                                  ), opt.dec_rnn_size)
        self.maxout = s2s.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.hidden_size = opt.dec_rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, ins_hidden, del_hidden, context, src_pad_mask, init_att, ins_pad_mask, del_pad_mask, que_hidden,que_len):
        emb = self.word_lut(input)

        self.attn2.applyMask(ins_pad_mask)
        wordEmb = self.word_lut(ins_hidden)
        ins_hidden = self.attn2(init_att,wordEmb.transpose(0, 1), None)[0]
        #ins_hidden = self.attn2(init_att,wordEmb.transpose(0, 1), None)[0]
        #ins_hidden = torch.sum(wordEmb, dim=0)
        self.attn2.applyMask(del_pad_mask)
        wordEmb = self.word_lut(del_hidden)
        #del_hidden = self.attn2(init_att,wordEmb.transpose(0, 1), None)[0]
        del_hidden = self.attn2(init_att,wordEmb.transpose(0, 1), None)[0]
        #del_hidden = torch.sum(wordEmb, dim=0)
        self.attn2.applyMask(que_len)
        wordEmb = self.word_lut(que_hidden)
        que_hidden =  self.attn2(init_att,wordEmb.transpose(0, 1), None)[0]



        g_outputs = []
        cur_context = init_att

        self.attn.applyMask(src_pad_mask)
        precompute = None
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context, ins_hidden, del_hidden,que_hidden], 1)
            output, hidden = self.rnn(input_emb, hidden)
            cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

            readout = self.readout(torch.cat((emb_t, output, cur_context, ins_hidden, del_hidden,que_hidden), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
        g_outputs = torch.stack(g_outputs)
        return g_outputs, hidden, attn, cur_context

class EditDecoder(nn.Module):
    def __init__(self, opt, dicts):
        self.opt = opt
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.enc_rnn_size + opt.word_vec_size * 2

        super(EditDecoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                     opt.word_vec_size,
                                     padding_idx=s2s.Constants.PAD)
        self.rnn = StackedGRU(opt.layers, input_size, opt.dec_rnn_size, opt.dropout)
        self.attn = s2s.modules.ConcatAttention(opt.enc_rnn_size, opt.dec_rnn_size, opt.att_vec_size)
        self.attn2 = s2s.modules.ConcatAttention(opt.word_vec_size, opt.enc_rnn_size,opt.word_vec_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.readout = nn.Linear((opt.enc_rnn_size
                                  + opt.dec_rnn_size
                                  + opt.word_vec_size
                                  + opt.word_vec_size * 2 # edit vector length
                                  ), opt.dec_rnn_size)
        self.maxout = s2s.modules.MaxOut(opt.maxout_pool_size)
        self.maxout_pool_size = opt.maxout_pool_size

        self.hidden_size = opt.dec_rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, ins_hidden, del_hidden, context, src_pad_mask, init_att):
        emb = self.word_lut(input)

        wordEmb = self.word_lut(ins_hidden)
        ins_hidden = self.attn2(context[-1],wordEmb.transpose(0, 1), None)[0]


        #ins_hidden = torch.sum(wordEmb, dim=0)
        wordEmb = self.word_lut(del_hidden)
        del_hidden = self.attn2(context[-1],wordEmb.transpose(0, 1), None)[0]
        #del_hidden = torch.sum(wordEmb, dim=0)



        g_outputs = []
        cur_context = init_att

        self.attn.applyMask(src_pad_mask)
        precompute = None
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            input_emb = emb_t
            if self.input_feed:
                input_emb = torch.cat([emb_t, cur_context, ins_hidden, del_hidden], 1)
            output, hidden = self.rnn(input_emb, hidden)
            cur_context, attn, precompute = self.attn(output, context.transpose(0, 1), precompute)

            readout = self.readout(torch.cat((emb_t, output, cur_context, ins_hidden, del_hidden), dim=1))
            maxout = self.maxout(readout)
            output = self.dropout(maxout)
            g_outputs += [output]
        g_outputs = torch.stack(g_outputs)
        return g_outputs, hidden, attn, cur_context
