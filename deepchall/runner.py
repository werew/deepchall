import json
import time
from .index import INDEX
from .backends.backend import ShapePlaceholder
from .nets.net import Net
from .langs.lang import Lang
from typing import Dict
from tqdm import tqdm, trange

class Runner:

    default_net_params = {}

    default_lang_params = {
        "max_samples": 100,
        "max_length": None, 
        "epochs": 10,
        "test_samples": 100,
    }

    default_backend_params = {}

    def __init__(self, config_path: str):
        with open(config_path) as fd:
            config = json.load(fd)

        self._check_config(config)

        # Init langs config
        self._langs_config = {}
        for name, params in config['langs'].items():
            # Merge params
            self._langs_config[name] = Runner.make_lang_params(
                lang = INDEX['langs'][params['lang']],
                user_params=params,
            )

        # Init nets config
        self._nets_config = {}
        for name, params in config['nets'].items():
            self._nets_config[name] = Runner.make_net_params(
                net = INDEX['nets'][params['net']],
                user_params = params,
            )

    @staticmethod
    def make_net_params(net: Net, user_params: Dict) -> Dict:
        # Merge params
        return {
            **Runner.default_net_params, # start by using default globals
            **Runner.unpack_extra_params(
                net.extra_params # put net-specific defaults
            ),
            **user_params, # config params take the precedence on everything
        }

    @staticmethod
    def make_lang_params(lang: Lang, user_params: Dict) -> Dict:
        # Merge params
        return {
            **Runner.default_lang_params, # start by using default globals
            **Runner.unpack_extra_params(
                lang.extra_params # put language-specific defaults
            ),
            **user_params, # config params take the precedence on everything
        }

    @staticmethod
    def unpack_extra_params(params: Dict) -> Dict:
        unpacked_params = {}
        for k,v in params.items():
            unpacked_params[k] = v[1]
        return unpacked_params

    def _check_config(self, config: Dict):
        if 'langs' not in config:
            raise ValueError('Missing langs key in config')
        if 'nets' not in config:
            raise ValueError('Missing nets key in config')
        if not isinstance(config['langs'], dict):
            raise ValueError('Expected a dict for langs key in config')
        if not isinstance(config['nets'], dict):
            raise ValueError('Expected a dict for nets key in config')

        for val in config['langs'].values():
            if val['lang'] not in INDEX['langs']:
                raise ValueError('Unknown language '+val['lang'])

        for val in config['nets'].values():
            if val['net'] not in INDEX['nets']:
                raise ValueError('Unknown net'+val['net'])

    # TODO: this function is insanely long, it needs some refactoring
    def _run_net(self, lang_name: str, net_name: str):

        # Get configs
        lang_config = self._langs_config[lang_name]
        net_config = self._nets_config[net_name]

        # Inititalize lang and get backend
        lang = INDEX['langs'][lang_config["lang"]]()
        lang.init(params=lang_config)
        bkd = lang.get()

        # Collect network initialization params
        net = INDEX['nets'][net_config["net"]]()
        net_params = {
            **lang_config,
            **net_config,
            'alphabet_size': lang.alphabet_size,
            'shape': lang.shape,
        }

        # Show lang parameters
        print("[*] Lang parameters:")
        for k, v in lang_config.items():
            print(f"\t{k}: {v}")

        # Show net parameters
        print("[*] Net parameters:")
        for k, v in net_params.items():
            print(f"\t{k}: {v}")

        # Some stats about the run
        stats = {
            'training_samples_generated' : 0,
            'max_training_sample_length' : 0,
            'min_training_sample_length' : 0,
            'training_samples_skipped' : 0,
            'training_samples_used' : 0,
            'correct_generated' : 0,
            'training_time': 0,
            'avg_length' : 0,
        }

        # Retrieve the length dimension
        try:
            length_index = lang.shape.index(None)
        except:
            length_index = None

        def _gen():
            max_length = lang_config['max_length']
            max_samples = lang_config['max_samples']
            gen = bkd.gen()
            pbar = tqdm(total=max_samples)
            while stats['training_samples_generated'] < max_samples:
                try:
                    sample = next(gen)
                    pbar.update(1)
                    stats['training_samples_generated'] += 1
                except StopIteration:
                    break

                if length_index is not None:
                    # Enforce max_length
                    if sample.shape[length_index] > max_length:
                        stats['training_samples_skipped'] += 1
                        continue
                    
                    # Update max_sample_length
                    if sample.shape[length_index] > stats['max_training_sample_length']:
                        stats['max_training_sample_length'] = sample.shape[length_index]

                    # Update min_sample_length
                    if sample.shape[length_index] < stats['min_training_sample_length']:
                        stats['min_training_sample_length'] = sample.shape[length_index]

                # Yield sample
                stats['training_samples_used'] += 1
                yield sample
            pbar.close()


        # Initialize and train the network
        net.init(params=net_params)
        print(f"[*] Start training")
        start_time = time.time()
        net.train(gen=_gen())
        end_time = time.time()
        stats['training_time'] = end_time - start_time
        print(f"[*] Training finished")

        print(f"[*] Start testing")
        sum_lengths = 0
        for _ in trange(lang_config['test_samples']):
            sample = net.gen()
            if bkd.parse(sample):
                stats['correct_generated'] += 1
            if length_index is not None:
                sum_lengths += sample.shape[length_index]

        if length_index is None:
            stats['avg_length'] = 'not applicable'
        else:
            stats['avg_length'] = sum_lengths / lang_config['test_samples']
        print(f"[*] Testing finished")
    
        return stats


    def run(self):
        for lang_name in self._langs_config.keys():
            for net_name in self._nets_config.keys():
                print(f"[*] Running lang: {lang_name} vs {net_name}")
                stats = self._run_net(lang_name, net_name)
                print(f"[*] Run finished")
                print(f"[*] Stats:")
                for k, v in stats.items():
                    print(f"\t{k}: {v}")


