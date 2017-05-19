"""
NLP helpers

* Text/Multitext can be used as batch generators for keras
* NAACL alignments reader
* Moses alignments writer
* AER

:Authors: - Wilker Aziz
"""
import numpy as np
import itertools
import os
import tempfile
from collections import Counter
import gzip
from io import TextIOWrapper


def smart_ropen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'rb'))
    else:
        return open(path, 'r')


def smart_wopen(path):
    """Opens files directly or through gzip depending on extension."""
    if path.endswith('.gz'):
        return TextIOWrapper(gzip.open(path, 'wb'))
    else:
        return open(path, 'w')


def read_naacl_alignments(path):
    """
    Read NAACL-formatted alignment files.

    :param path: path to file
    :return: a list of pairs [sure set, possible set]
        each entry in the set maps an input position to an output position
        sentences start from 1 and a NULL token is indicated with position 0
    """
    with smart_ropen(path) as fi:
        ainfo = {}
        for i, line in enumerate(fi.readlines()):
            fields = line.split()
            if not fields:
                continue
            sure = True  # by default we assumed Sure links
            prob = 1.0  # by default we assume prob 1.0
            if len(fields) < 3:
                raise ValueError('Missing required fields in line %d: %s' % (i, line.strip()))
            snt_id, x, y = int(fields[0]), int(fields[1]), int(fields[2])
            if len(fields) == 5:
                sure = fields[3] == 'S'
                prob = float(fields[4])
            if len(fields) == 4:
                if fields[3] in {'S', 'P'}:
                    sure = fields[3] == 'S'
                else:
                    prob = float(fields[3])
            snt_info = ainfo.get(snt_id, None)
            if snt_info is None:
                snt_info = [set(), set()]  # S and P sets
                ainfo[snt_id] = snt_info
            if sure:  # Note that S links are also P links: http://dl.acm.org/citation.cfm?id=992810
                snt_info[0].add((x, y))
                snt_info[1].add((x, y))
            else:
                snt_info[1].add((x, y))
    return tuple(v for k, v in sorted(ainfo.items(), key=lambda pair: pair[0]))


def save_moses_alignments(alignments, posteriors, lengths, ostream):
    """
    Save viterbi alignments in Moses format.

    :param alignments: (B, N) matrix of alignments
    :param posteriors: (B, N) matrix of posterior alignment probability
    :param lengths: (B,) matrix of target length
    :param ostream: output stream
    """
    for a, p, l in zip(alignments, posteriors, lengths):
        print(' '.join('{0}-{1}|{2:.4f}'.format(a[j], j + 1, p[j]) for j in range(l)), file=ostream)


class AERSufficientStatistics:
    """
    Object used to compute AER over for a corpus.
    """

    def __init__(self):
        self.a_and_s = 0
        self.a_and_p = 0
        self.a = 0
        self.s = 0

    def __str__(self):
        return '%s/%s/%s/%s %s' % (self.a_and_s, self.a_and_p, self.a, self.s, self.aer())

    def update(self, sure, probable, predicted):
        """
        Update AER sufficient statistics for a set of predicted links given goldstandard information.

        :param sure: set of sure links
        :param probable: set of probable links (must incude sure links)
        :param predicted: set of predicted links
        """
        self.a_and_s += len(predicted & sure)
        self.a_and_p += len(predicted & probable)
        self.a += len(predicted)
        self.s += len(sure)

    def aer(self):
        """Return alignment error rate: 1 - (|A & S| + |A & P|)/(|A| + |S|)"""
        return 1 - (self.a_and_s + self.a_and_p) / (self.a + self.s)


class Tokenizer:
    """
    A Tokenizer splits streams of text into tokens (on white spaces) and builds and internal vocabulary.
     The vocabulary can be bounded in size and it contains some standard symbols:
        Required:
            * pad_str: string used for padding (its value is mostly useless, but its id is important, it gets id 0)
            * unk_str: string used to map an unknown symbol in case of a bounded vocabulary (it gets id 1)
        Optional:
            * bos_str: string added to the beginning of every sequence (before padding)
            * eos_str: string added to the end of every sequence (before padding)
        To bound a vocabulary set nb_words to a positive integer. This will cap the number of words in the vocabulary,
        but the total vocabulary size will include at least a few more tokens (pad_str and unk_str and possibly boundary
        symbols if configured).

    You can use a collection of copora to fit a tokenizer and then convert one by one into sequences of integers.

    """

    def __init__(self, nb_words=None, bos_str=None, eos_str=None, unk_str='-UNK-', pad_str='-PAD-'):
        """

        :param nb_words: if not None, keeps only the most frequent tokens
        :param bos_str: an optional BOS symbol
        :param eos_str: an optional EOS symbol
        :param unk_str: a string to map UNK tokens to
        :param pad_str: a string to visualise padding
        """
        self._nb_words = nb_words
        self._counts = Counter()
        self._vocab = {pad_str: 0, unk_str: 1}
        self._tokens = [pad_str, unk_str]
        self._pad_str = pad_str
        self._unk_str = unk_str
        self._bos_str = bos_str
        self._eos_str = eos_str
        if bos_str is not None:
            self._vocab[bos_str] = len(self._tokens)
            self._tokens.append(bos_str)
        if eos_str is not None:
            self._vocab[eos_str] = len(self._tokens)
            self._tokens.append(eos_str)

        if self._bos_str and self._eos_str:
            self._tokenize = lambda s: [self._bos_str] + s.split() + [self._eos_str]
        elif self._bos_str:
            self._tokenize = lambda s: [self._bos_str] + s.split()
        elif self._eos_str:
            self._tokenize = lambda s: s.split() + [self._eos_str]
        else:
            self._tokenize = lambda s: s.split()

    def fit_one(self, input_stream):
        """
        This method fits the tokenizer to a corpus read off a single input stream.

        :param input_stream: an iterable of strings
        """
        self.fit_many([input_stream])

    def fit_many(self, input_streams):
        """
        This method fits the tokenizer to a collection of texts.
        Each text is read off an input stream.

        :param input_streams: a collection of input streams (e.g. list of file handlers)
        """
        for stream in input_streams:
            for line in stream:
                self._counts.update(line.split())

        for token, count in self._counts.most_common(self._nb_words):
            self._vocab[token] = len(self._tokens)
            self._tokens.append(token)

    def vocab_size(self):
        return len(self._tokens)

    def to_str(self, token_id):
        return self._tokens[token_id]

    def to_sequences(self, input_stream, dtype='int64'):
        unk_id = self._vocab[self._unk_str]
        return [np.array([self._vocab.get(word, unk_id) for word in self._tokenize(line)], dtype=dtype)
                         for line in input_stream]

    def to_sequences_iterator(self, input_stream, dtype='int64'):
        unk_id = self._vocab[self._unk_str]
        for line in input_stream:
            yield np.array([self._vocab.get(word, unk_id) for word in self._tokenize(line)], dtype=dtype)

    def write_as_text(self, matrix, output_stream, skip_pad=True):
        """
        Write the elements of a matrix in text format (translating integers back to words) to an output stream.

        :param matrix: samples
        :param output_stream: where to write text sequences to
        :param skip_pad: whether or not pads should be ignored (defaults to True)
        """
        if skip_pad:
            for seq in matrix:
                print(' '.join(self._tokens[tid] for tid in itertools.takewhile(lambda x: x != 0, seq)),
                      file=output_stream)
        else:
            for seq in matrix:
                print(' '.join(self._tokens[tid] for tid in seq), file=output_stream)


def bound_length(input_paths, tokenizers, shortest, longest):
    """
    Return an np.array which flags whether all parallel segments comply with length constraints
    and count the number of tokens in each stream (considering valid sequences only).

    :param input_paths: paths (list/tuple) to each part of the parallel collection
    :param tokenizers: list/tuple of tokenizers
    :param shortest: shortest valid sequence for each part of the parallel collection
    :param longest: longest valid sequence for each part of the parallel collection
    :return: selection (nb_samples,) and counts (nb_streams,)
    """

    # get an iterator for each stream
    nb_streams = len(input_paths)
    iterators = [tokenizers[i].to_sequences_iterator(smart_ropen(input_paths[i])) for i in range(nb_streams)]

    # length checks
    selection = []
    nb_tokens = [0] * nb_streams
    for seqs in zip(*iterators):  # get a sequence from each iterator
        # check if every sequence complies with its respective length bounds
        if not all(lower <= seq.shape[0] <= upper for lower, upper, seq in zip(shortest, longest, seqs)):
            selection.append(False)  # excluded
        else:
            selection.append(True)  # included
            # increase token count
            for i, seq in enumerate(seqs):
                nb_tokens[i] += seq.shape[0]

    return np.array(selection, dtype=bool), np.array(nb_tokens, dtype='int64')


def construct_mmap(input_path, output_path, tokenizer, selection, nb_tokens, dtype):
    """
    Construct memory map for selected sentences in a corpus.

    :param input_path: path to text
    :param output_path: path to memory map file
    :param tokenizer: tokenizer for text
    :param selection: array of binary selectors
    :param nb_tokens: total number of tokens in the selected corpus
    :param dtype: data type for memmap
    :return: np.array with shape (nb_samples,) where array[i] is the length of the ith sequence
    """

    # construct memory mapped array
    mmap = np.memmap(output_path, dtype=dtype, mode='w+', shape=nb_tokens)

    # prepare for populating memmap
    offset = 0
    sample_length = []

    # populate memory map
    for sid, seq in enumerate(tokenizer.to_sequences_iterator(smart_ropen(input_path))):
        if not selection[sid]:  # skip sentences that do not comply with length constraints
            continue
        # here we have a valid sequence, thus we memory map it
        mmap[offset:offset + seq.shape[0]] = seq
        offset += seq.shape[0]
        sample_length.append(seq.shape[0])

    del mmap

    return np.array(sample_length, dtype='int64')


class Text:
    """
    This class is used to represent large text collections as a matrix of integers.

    It uses a pre-trained Tokenizer and it can impose a limit on sentence length.
    It uses memory mapped files for memory efficiency,
     and it provides a generator for batches of a given size. This generator may iterate once through the data
     or indefinitely in an endless cycle.

    TODO: reload memmap when possible (I find this a bit dangerous though since several options affect its structure)

    """

    MASK = 0
    TRIM = 1
    COMPLETE = 2
    DISCARD = 3
    STRATEGY_MAP = {'mask': MASK, 'trim': TRIM, 'complete': COMPLETE, 'discard': DISCARD}

    def __init__(self, input_path, tokenizer: Tokenizer,
                 shortest=1,
                 longest=np.inf,
                 trim=False,
                 output_dir=None,
                 tmp_dir=None,
                 batch_dtype='int64',
                 mask_dtype='float32',
                 name='text',
                 selection=None,
                 nb_tokens=None):
        """
        Wrap a corpus for string->integer conversion.

        An object of this class cleans up after itself: randomly generated files created by this class
            are removed on destruction. Note that, if a user specifies output_dir,
            then the the memory map will be deleted.

        :param input_path: path to a file containing the raw text
        :param tokenizer: a Tokenizer to turn text sequences into integer sequences
        :param shortest: the length of the shortest valid sequence (defaults to 1 which is also the minimum)
        :param longest: the length of the longest valid sentence (defaults to inf)
        :param trim: trim batches to the longest sentence in the corpus (defaults to False)
            but longest=np.inf causes trim to be overwritten to True
        :param output_dir: where to store the memory map (defaults to None in which case tmp_dir will be used)
        :param tmp_dir: a temporary directory used in case output_dir is None (defaults to None in which case a
            the system's tmp space will be used)
        :param batch_dtype: data type for batches
        :param mask_dtype: data type for masks
        :param name: name of the corpus (file will end in .dat)
            * if the memory map lives in output_dir then its file name will be '{}.dat'.format(name)
            * if the memory map lives in temp_dir then its file name will be obtained with
                tempfile.mkstemp(prefix=name, suffix='.dat', dir=tmp_dir, text=False)
            in this case, the file will be deleted when the Text object gets destructed
        :param selection: uses a subset of the data specified through a np.array with a binary selector per sample
        :param nb_tokens: total number of tokens in the selection
            selection and nb_tokens are used when multiple texts are simultaneously constrained for length
            users probably would never need to specify these variables by hand
        """
        assert shortest > 0, '0-length sequences are not such a great idea'
        if longest == np.inf:  # overwrites trim when longest is np.inf
            trim = True

        self._input_path = input_path
        self._tokenizer = tokenizer
        self._batch_dtype = batch_dtype
        self._mask_dtype = mask_dtype
        self._to_remove = {}

        # create a file to store the corpus
        if output_dir is None:
            if tmp_dir:
                tmp_dir = os.path.abspath(tmp_dir)  # make it absolute
                os.makedirs(tmp_dir, exist_ok=True)  # make sure it exists
                _, memmap_path = tempfile.mkstemp(prefix=name, suffix='.dat', dir=tmp_dir, text=False)
            else:
                _, memmap_path = tempfile.mkstemp(prefix=name, dir=tmp_dir, text=False)  # create a random file name
            self._to_remove['memmap'] = memmap_path  # mark for deletion (since this lives in a temporary directory)
        else:  # user chose an output (not temporary) directory
            output_dir = os.path.abspath(output_dir)  # make it absolute
            os.makedirs(output_dir, exist_ok=True)  # make sure it exists
            memmap_path = os.path.join(output_dir, '{}.dat'.format(name))
            # do not schedule deletion (user probably wants to keep folder and/or file)
        self._memmap_path = memmap_path

        if selection is None or nb_tokens is None:
            # bound sequences for length and count number of resulting tokens
            selection, nb_tokens = bound_length([input_path], [tokenizer], [shortest], [longest])
            nb_tokens = nb_tokens[0]  # bound_length returns a list with a total per stream
        # construct mmap given length constraints (expressed through selection and nb_tokens)
        self._sample_length = construct_mmap(input_path, memmap_path, tokenizer,
                                             selection, nb_tokens, dtype=batch_dtype)
        # total number of tokens
        self._nb_tokens = nb_tokens
        assert nb_tokens == self._sample_length.sum(), 'Something went wrong: input nb_tokens != memmaped nb_tokens'
        self._nb_samples = len(self._sample_length)
        # longest sequence in corpus (possibly not trimmed)
        self._longest = longest if not trim else self._sample_length.max()
        self._selection = selection
        self._name = name

    @property
    def name(self):
        return self._name

    def __del__(self):
        if 'memmap' in self._to_remove:
            try:
                os.unlink(self._to_remove['memmap'])
            except FileNotFoundError:
                pass
        if 'tmp_dir' in self._to_remove:
            try:
                os.rmdir(self._to_remove['tmp_dir'])
            except FileNotFoundError:  # the directory somehow disappeared
                pass
            except OSError:  # probably there's more stuff in the directory
                pass

    def iter_selection_flags(self):
        """Iterate over the selection flags"""
        return iter(self._selection)

    def nb_streams(self):
        """A Text is a single stream"""
        return 1

    def memmap_path(self, stream=0):
        """Where the memory map is stored"""
        return self._memmap_path

    def nb_samples(self):
        """Total number of sequences in the corpus"""
        return self._nb_samples

    def nb_tokens(self, stream=0):
        """Total number of tokens in the corpus"""
        return self._nb_tokens

    def vocab_size(self, stream=0):
        """Size of the vocabulary (including -PAD-, -UNK-, and other special symbols)"""
        return self._tokenizer.vocab_size()

    def longest_sequence(self, stream=0):
        """Length of the longest sequence in the corpus"""
        return self._longest

    def batch_iterator(self, batch_size, endless=False, shorter_batch='mask', dynamic_sequence_length=False):
        """
        Iterate over an input stream yielding batches of a certain size.

        :param batch_size: number of samples/sequences in the batch
        :param endless: cycle endlessly over the samples in the corpus (defaults to False)
        :param shorter_batch: strategy to deal with a shorter batch at the end of the corpus
            * 'mask': masks missing sequences in last batch
            * 'trim': truncates the last batch (implies dynamic number of samples per batch)
            * 'complete': loops over to the beginning of the corpus gathering samples to complete the batch
            * 'discard': ditches the last batch
            * anything else will silently get mapped to 'mask'
        :param dynamic_sequence_length: with dynamic sequence length with trim columns as to fit the longest
            sample in the batch (default to False)
        :return: generator of pairs (batch, mask)
        """
        mmap = np.memmap(self._memmap_path, dtype=self._batch_dtype, mode='r')

        nb_total_samples = self.nb_samples()

        endless_iterator = itertools.cycle(enumerate(self._sample_length))
        offset = 0
        n_cols = self.longest_sequence()  # by default batch and mask are created with as many columns as necessary

        # configure length strategy
        if dynamic_sequence_length:
            trim_length = lambda pair, longest: (pair[0][:, :longest], pair[1][:, :longest])
        else:
            trim_length = lambda pair, longest: pair

        # configure shorter batch strategy
        shorter_batch = Text.STRATEGY_MAP.get(shorter_batch, Text.MASK)
        if shorter_batch == Text.TRIM:
            trim_size = lambda pair, size: (pair[0][:size, :], pair[1][:size, :])
        else:
            trim_size = lambda pair, size: pair

        # Generate batches and masks potentially indefinitely
        generating = True
        while generating:
            batch = np.zeros((batch_size, n_cols), dtype=self._batch_dtype)
            mask = np.zeros((batch_size, n_cols), dtype=self._mask_dtype)
            valid_batch = True
            longest_in_batch = 0
            samples_in_batch = 0
            for row in range(batch_size):
                seq_id, sample_length = next(endless_iterator)  # get the next length
                if seq_id == 0:  # we are back at the beginning of the corpus
                    offset = 0
                batch[row, :sample_length] = mmap[offset: offset + sample_length]
                mask[row, :sample_length] = 1
                offset += sample_length
                # update tightest possible shape
                longest_in_batch = max(longest_in_batch, sample_length)
                samples_in_batch += 1
                # If we hit the 0-based end of the corpus
                #  and this is a shorter batch (we are not at the 0-based end of the batch)
                if seq_id + 1 == nb_total_samples and row + 1 < batch_size:
                    if not endless:  # this is the last batch we may generate
                        generating = False
                    if shorter_batch == Text.COMPLETE:  # we will generate a complete batch
                        # thus just go on (next samples will come from the beginning of the corpus)
                        continue
                    if shorter_batch == Text.DISCARD:  # we are not yielding this batch
                        # thus we invalidate it
                        valid_batch = False
                    break  # DISCARD/TRIM/MASK all lead to a break for this batch

            if valid_batch:  # here we yield a batch after attempting trimming rows and columns
                yield trim_size(trim_length((batch, mask), longest_in_batch), samples_in_batch)


class Multitext:
    """
    This class wraps a collection of parallel Text objects.

    It extends the functionality of Text by allowing parallel streams
    """

    def __init__(self, input_paths: tuple, tokenizers: tuple,
                 shortest=None,
                 longest=None,
                 trim=None,
                 output_dir=None,
                 tmp_dir=None,
                 batch_dtype='int64',
                 mask_dtype='float32',
                 name='bitext',
                 selection=None,
                 nb_tokens=None):
        """
        Wraps a collection of Text objects, one per stream (check Text's note on cleanup).


        :param input_paths: path to each half of the parallel corpus
        :param tokenizers: a Tokenizer for each half of the parallel corpus
        :param shortest: a pair specifying the length of the shortest valid sequence (defaults to 1 for all streams)
        :param longest: a pair specifying the length of the longest valid sentence (defaults to inf for all streams)
        :param trim: a pair specifying whther to trim batches to the longest sentence in the corpus
            defaults to False for all streams, but if longest is unbounded, trim will be overwritten to True
        :param output_dir: where to store the memory map (defaults to None in which case tmp_dir will be used)
        :param tmp_dir: a temporary directory used in case output_dir is None (defaults to None in which case a
            the system's tmp space will be used)
        :param batch_dtype: data type for batches
        :param mask_dtype: data type for masks
        :param name: name of the corpus (file will end in .dat)
            * if memory maps live in output_dir then each file name will be '{}-{}.dat'.format(name, stream_nb)
            * if memory maps live in temp_dir then each file name will be obtained with
                tempfile.mkstemp(prefix='{}-{}'.format(name, stream_nb), suffix='.dat', dir=tmp_dir, text=False)
            in this case, files will be deleted when the Text objects get destructed
        :param selection: uses a subset of the data specified through a np.array with a binary selector per sample
            Multitext can figure this out by itself.
        :param nb_tokens: total number of tokens in the selection
            selection and nb_tokens are used when multiple texts are simultaneously constrained for length
            users probably would never need to specify these variables by hand
        """
        nb_streams = len(input_paths)

        # normalise some default attributes
        if shortest is None:
            shortest = [1] * nb_streams
        if longest is None:
            longest = [np.inf] * nb_streams
        if trim is None:
            trim = [False] * nb_streams

        assert all(lower > 0 for lower in shortest), '0-length sequences are not such a great idea'
        assert len(input_paths) == len(tokenizers) == len(shortest) == len(longest) == len(trim) == nb_streams, \
            'Be consistent wrt input/tokenizers/shortest/longest: I expect %d input streams' % nb_streams

        if selection is None or nb_tokens is None:
            # select parallel sentences complying with length constraints
            selection, nb_tokens = bound_length(input_paths, tokenizers, shortest, longest)

        corpora = []  # type: list[Text]
        for i in range(nb_streams):
            corpora.append(Text(input_path=input_paths[i],
                                tokenizer=tokenizers[i],
                                shortest=shortest[i],
                                longest=longest[i],
                                trim=trim[i],
                                output_dir=output_dir,
                                tmp_dir=tmp_dir,
                                batch_dtype=batch_dtype,
                                mask_dtype=mask_dtype,
                                selection=selection,
                                nb_tokens=nb_tokens[i],
                                name='{}-{}'.format(name, i)))

        self._corpora = tuple(corpora)  # type: tuple[Text]
        self._nb_samples = selection.sum()
        self._batch_dtype = batch_dtype
        self._mask_dtype = mask_dtype
        self._selection = selection
        self._name = name

    @property
    def name(self):
        return self._name

    def iter_selection_flags(self):
        """Iterate over the selection flags"""
        return iter(self._selection)

    def nb_streams(self):
        return len(self._corpora)

    def memmap_path(self, stream):
        return self._corpora[stream].memmap_path()

    def nb_tokens(self, stream):
        """Total number of tokens in the corpus"""
        return self._corpora[stream].nb_tokens()

    def vocab_size(self, stream):
        """Size of the vocabulary (including -PAD-, -UNK-, and other special symbols)"""
        return self._corpora[stream].vocab_size()

    def nb_samples(self):
        """Total number of sequences in the corpus"""
        return self._nb_samples

    def longest_sequence(self, stream):
        """Length of the longest sequence in the corpus"""
        return self._corpora[stream].longest_sequence()

    def batch_iterator(self, batch_size, endless=False, shorter_batch='mask', dynamic_sequence_length=False):
        """
        Iterate over an input stream yielding batches of a certain size.

        :param batch_size: number of samples/sequences in the batch
        :param endless: cycle endlessly over the samples in the corpus (defaults to False)
        :param shorter_batch: strategy to deal with a shorter batch at the end of the corpus
            * 'mask': masks missing sequences in last batch
            * 'trim': truncates the last batch (implies dynamic number of samples per batch)
            * 'complete': loops over to the beginning of the corpus gathering samples to complete the batch
            * 'discard': ditches the last batch
            * anything else will silently get mapped to 'mask'
        :param dynamic_sequence_length: with dynamic sequence length with trim columns as to fit the longest
            sample in the batch (default to False)
        :return: generator of pairs (batch, mask), one pair per stream
        """

        iterators = [corpus.batch_iterator(batch_size, endless, shorter_batch, dynamic_sequence_length)
                     for corpus in self._corpora]

        while True:  # because this is a generator, we leave the loop with a StopIteration exception
            yield [next(iterator) for iterator in iterators]


def test_text(input_path, output_path):
    """
    Test the reconstruction of a corpus passing it through Tokenizer/Text pipeline.
        Example:
            text.test_text('data/en-fr/test.en-fr.en', 'data/en-fr/test.en-fr.en-mono')

    :param input_path: a text file
    :param output_path: where to save its reconstruction
    """
    tk = Tokenizer()
    tk.fit_one(smart_ropen(input_path))
    text = Text(input_path, tk)
    with open(output_path, 'w') as fo:
        for b, m in text.batch_iterator(100, shorter_batch='trim'):
            tk.write_as_text(b, fo)
    return text


def test_bitext(input_path1, input_path2, output_path1, output_path2):
    """
    Test the reconstruction of a bilingual corpus passing it through Tokenizer/Multitext pipeline.

        Example:
            text.test_bitext('data/en-fr/test.en-fr.en', 'data/en-fr/test.en-fr.fr', 'data/en-fr/test.en-fr.en-bi', 'data/en-fr/test.en-fr.fr-bi')

    :param input_path1: a text file
    :param input_path2: a parallel text file
    :param output_path1: where to save the reconstruction of the first stream
    :param output_path2: where to save the reconstruction of the second stream
    """
    tk1 = Tokenizer()
    tk2 = Tokenizer()
    tk1.fit_one(open(input_path1))
    tk2.fit_one(open(input_path2))

    bitext = Multitext([input_path1, input_path2], [tk1, tk2])
    with open(output_path1, 'w') as fo1:
        with open(output_path2, 'w') as fo2:
            for (b1, m1), (b2, m2) in bitext.batch_iterator(100, shorter_batch='trim'):
                tk1.write_as_text(b1, fo1)
                tk2.write_as_text(b2, fo2)
    return bitext


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print('Usage: %s input.src input.tgt output.src output.tgt' % sys.argv[0])
        sys.exit()
    test_bitext(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
