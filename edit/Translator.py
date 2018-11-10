import s2s
import torch.nn as nn
import torch
from torch.autograd import Variable

try:
    import ipdb
except ImportError:
    pass


class Translator(object):
    def __init__(self, opt, model=None, dataset=None):
        self.opt = opt

        if model is None:

            checkpoint = torch.load(opt.model,map_location=lambda storage, loc: storage)

            model_opt = checkpoint['opt']
            self.src_dict = checkpoint['dicts']['src']
            self.tgt_dict = checkpoint['dicts']['tgt']

            self.word_vec_size = model_opt.word_vec_size
            self.enc_rnn_size = model_opt.enc_rnn_size
            self.dec_rnn_size = model_opt.dec_rnn_size

            encoder = s2s.Models.Encoder(model_opt, self.src_dict)
            editEncoer = s2s.EditModels.EditEncoder(model_opt, self.src_dict)
            decoder = s2s.EditModels.EditAttDecoder(model_opt, self.tgt_dict)
            decIniter = s2s.Models.DecInit(model_opt)

            model = s2s.EditModels.IDEditModel(encoder, editEncoer, decoder, decIniter)

            generator = nn.Sequential(
                nn.Linear(model_opt.dec_rnn_size // model_opt.maxout_pool_size, self.tgt_dict.size()),
                nn.LogSoftmax())

            model.load_state_dict(checkpoint['model'])
            generator.load_state_dict(checkpoint['generator'])

            if opt.cuda:
                model.cuda()
                generator.cuda()
            else:
                model.cpu()
                generator.cpu()

            model.generator = generator
        else:
            self.src_dict = dataset['dicts']['src']
            self.tgt_dict = dataset['dicts']['tgt']

            self.enc_rnn_size = opt.enc_rnn_size
            self.dec_rnn_size = opt.dec_rnn_size
            self.opt.cuda = True if len(opt.gpus) >= 1 else False
            self.opt.n_best = 1
            self.opt.replace_unk = False

        self.tt = torch.cuda if opt.cuda else torch
        self.model = model
        self.model.eval()

        self.copyCount = 0

    def buildData(self, srcBatch, srcInsBatch, srcDelBatch, goldBatch):
        srcData = [self.src_dict.convertToIdx(b,
                                              s2s.Constants.UNK_WORD) for b in srcBatch]
        srcInsData = [self.tgt_dict.convertToIdx(b,
                                              s2s.Constants.UNK_WORD) for b in srcInsBatch]
        srcDelData = [self.tgt_dict.convertToIdx(b,
                                              s2s.Constants.UNK_WORD) for b in srcDelBatch]
        tgtData = None
        if goldBatch:
            tgtData = [self.tgt_dict.convertToIdx(b,
                                                  s2s.Constants.UNK_WORD,
                                                  s2s.Constants.BOS_WORD,
                                                  s2s.Constants.EOS_WORD) for b in goldBatch]

        return s2s.IDDataSet(srcData, srcInsData, srcDelData, tgtData, self.opt.batch_size, self.opt.cuda)

    def buildTargetTokens(self, pred, src, attn):
        pred_word_ids = [x.item() for x in pred]
        tokens = self.tgt_dict.convertToLabels(pred_word_ids, s2s.Constants.EOS)
        tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == s2s.Constants.UNK_WORD:
                    _, maxIndex = attn[i].max(0)
                    tokens[i] = src[maxIndex[0]]
        return tokens

    def translateBatch(self, srcBatch, srcInsBatch, srcDelBatch, tgtBatch):
        batchSize = srcBatch[0].size(1)
        beamSize = self.opt.beam_size

        #  (1) run the encoder on the src
        encStates, context = self.model.encoder(srcBatch)
        srcBatch = srcBatch[0]  # drop the lengths needed for encoder
        # enc_ins_hidden = self.model.editEncoder(srcInsBatch).data.repeat(beamSize, 1)
        # enc_del_hidden = self.model.editEncoder(srcDelBatch).data.repeat(beamSize, 1)

        enc_ins_hidden = srcInsBatch[0].data.repeat(1, beamSize)
        enc_del_hidden = srcDelBatch[0].data.repeat(1, beamSize)

        decStates = self.model.decIniter(encStates[1])  # batch, dec_hidden

        #  (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = context.data.repeat(1, beamSize, 1)
        decStates = decStates.unsqueeze(0).data.repeat(1, beamSize, 1)
        att_vec = self.model.make_init_att(context)
        padMask = srcBatch.data.eq(s2s.Constants.PAD).transpose(0, 1).unsqueeze(0).repeat(beamSize, 1, 1).float()
        insMask = srcInsBatch[0].data.repeat(1, beamSize).eq(s2s.Constants.PAD).transpose(0, 1).float()
        delMask = srcDelBatch[0].data.repeat(1, beamSize).eq(s2s.Constants.PAD).transpose(0, 1).float()

        beam = [s2s.Beam(beamSize, self.opt.cuda) for k in range(batchSize)]
        batchIdx = list(range(batchSize))
        remainingSents = batchSize

        for i in range(self.opt.max_sent_length):
            # Prepare decoder input.
            input = torch.stack([b.getCurrentState() for b in beam
                                 if not b.done]).transpose(0, 1).contiguous().view(1, -1)
            #print(enc_ins_hidden.shape,input.shape,insMask.shape,padMask.shape)
            g_outputs, decStates, attn, att_vec = self.model.decoder(input, decStates,
                                                                     enc_ins_hidden,
                                                                     enc_del_hidden,
                                                                     context,
                                                                     padMask.view(-1, padMask.size(2)), att_vec,insMask,delMask)

            # g_outputs: 1 x (beam*batch) x numWords
            g_outputs = g_outputs.squeeze(0)
            g_out_prob = self.model.generator.forward(g_outputs)

            # batch x beam x numWords
            wordLk = g_out_prob.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()
            attn = attn.view(beamSize, remainingSents, -1).transpose(0, 1).contiguous()

            active = []
            father_idx = []
            for b in range(batchSize):
                if beam[b].done:
                    continue

                idx = batchIdx[b]
                if not beam[b].advance(wordLk.data[idx], attn.data[idx]):
                    active += [b]
                    father_idx.append(beam[b].prevKs[-1])  # this is very annoying

            if not active:
                break

            # to get the real father index
            real_father_idx = []
            for kk, idx in enumerate(father_idx):
                real_father_idx.append(idx * len(father_idx) + kk)

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            activeIdx = self.tt.LongTensor([batchIdx[k] for k in active])
            batchIdx = {beam: idx for idx, beam in enumerate(active)}
            def updateActiveIns(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(rnnSize, -1, remainingSents)
                newSize = list(t.size())
                newSize[-1] = newSize[-1] * len(activeIdx) // remainingSents
                return view.index_select(2, activeIdx).view(*newSize)

            def updateActive(t, rnnSize):
                # select only the remaining active sentences
                view = t.data.view(-1, remainingSents, rnnSize)
                newSize = list(t.size())
                newSize[-2] = newSize[-2] * len(activeIdx) // remainingSents
                return view.index_select(1, activeIdx).view(*newSize)

            # def updateActiveIns(t, rnnSize):
            #     # select only the remaining active sentences
            #     view = t.data.view(-1, remainingSents, rnnSize)
            #     newSize = list(t.size())
            #     newSize[-1] = newSize[-1] * len(activeIdx) // remainingSents
            #     return view.index_select(1, activeIdx).view(*newSize)

            decStates = updateActive(decStates, self.dec_rnn_size)
            context = updateActive(context, self.enc_rnn_size)
            att_vec = updateActive(att_vec, self.enc_rnn_size)
            enc_ins_hidden = updateActiveIns(enc_ins_hidden, enc_ins_hidden.size()[0])
            enc_del_hidden = updateActiveIns(enc_del_hidden, enc_del_hidden.size()[0])
            padMask = padMask.index_select(1, activeIdx)
            insMask = enc_ins_hidden.eq(s2s.Constants.PAD).transpose(0, 1).float()
            delMask = enc_del_hidden.eq(s2s.Constants.PAD).transpose(0, 1).float()
            #print(insMask.shape)

            # set correct state for beam search
            previous_index = torch.stack(real_father_idx).transpose(0, 1).contiguous()
            decStates = decStates.view(-1, decStates.size(2)).index_select(0, previous_index.view(-1)).view(
                *decStates.size())
            att_vec = att_vec.view(-1, att_vec.size(1)).index_select(0, previous_index.view(-1)).view(*att_vec.size())

            remainingSents = len(active)

        # (4) package everything up
        allHyp, allScores, allAttn = [], [], []
        n_best = self.opt.n_best

        for b in range(batchSize):
            scores, ks = beam[b].sortBest()

            allScores += [scores[:n_best]]
            valid_attn = srcBatch.data[:, b].ne(s2s.Constants.PAD).nonzero().squeeze(1)
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            attn = [a.index_select(1, valid_attn) for a in attn]
            allHyp += [hyps]
            allAttn += [attn]

        return allHyp, allScores, allAttn, None

    def translate(self, srcBatch, srcInsBatch, srcDelBatch, goldBatch):
        #  (1) convert words to indexes
        print(len(srcBatch))
        dataset = self.buildData(srcBatch, srcInsBatch, srcDelBatch, goldBatch)

        # (wrap(srcBatch), lengths), (wrap(srcInsBatch),), (wrap(srcDelBatch),), (wrap(tgtBatch),), indices
        src, srcIns, srcDel, tgt, indices = dataset[0]

        #  (2) translate
        pred, predScore, attn, _ = self.translateBatch(src, srcIns, srcDel, tgt)
        pred, predScore, attn = list(zip(
            *sorted(zip(pred, predScore, attn, indices),
                    key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        predBatch = []
        for b in range(src[0].size(1)):
            predBatch.append(
                [self.buildTargetTokens(pred[b][n], srcBatch[b], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return predBatch, predScore, None
