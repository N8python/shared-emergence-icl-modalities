"""Regression checks for faithful recovery of displayed sampled-error inputs."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sampler', ROOT/'scripts/analysis/sample_mistakes_t128.py')
sampler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampler)


class InputRecoveryTests(unittest.TestCase):
    def test_invariant_function_requires_demonstrations_not_only_query(self):
        # Eight-bit parity is unchanged by complementing every input bit.
        # Either query orientation fits the gold answer, but only one decodes
        # the clean demonstration correctly. The model's wrong prediction
        # must never participate in choosing that orientation.
        program = ('parity_fill',)
        query = '00110011'
        example = '00000001'
        encode = lambda s: s.translate(str.maketrans('01', 'AC'))
        trial = {'query': encode(query), 'expected': '00000000',
                 'prediction': '11111111',
                 'prompt': encode(example)+encode('11111111')+'T'+encode(query)}
        self.assertEqual(sampler.dsl.run_program(program,query),
                         sampler.dsl.run_program(program,sampler.dsl.flip_bits(query)))
        recovered, audit = sampler.recover_input('evo2',program,trial,1)
        self.assertEqual(recovered,query)
        self.assertEqual(audit['symbol_to_bit'],{'A':'0','C':'1'})
        trial['prediction'] = ''
        self.assertEqual(sampler.recover_input('evo2',program,trial,1)[0],query)

    def test_all_one_symbol_query_is_recovered_using_prompt_alphabet(self):
        trial = {'query':'AAAAAAAA','expected':'00000000',
                 'prompt':'AAAAAAACAAAAAAACTAAAAAAAA'}
        self.assertEqual(sampler.recover_input('evo2',('identity',),trial,1)[0],'00000000')

    def test_inconsistent_gold_query_is_rejected(self):
        trial = {'query':'AAAAAAAA','expected':'01010101',
                 'prompt':'AAAAAAACAAAAAAACTAAAAAAAA'}
        with self.assertRaises(AssertionError):
            sampler.recover_input('evo2',('identity',),trial,1)

    def test_protein_start_token_and_nextterm_commas(self):
        expected = '10101010'
        for run,prompt in [('progen2','1AAAAAAACCAAAAAAATACACACAC'),
                           ('nextterm','AAAAAAAC,CAAAAAAA,ACACACAC,')]:
            trial={'query':'ACACACAC','expected':expected,'prompt':prompt}
            with self.subTest(run=run):
                self.assertEqual(sampler.recover_input(run,('reverse_bits',),trial,1)[0],'01010101')


if __name__ == '__main__':
    unittest.main()
