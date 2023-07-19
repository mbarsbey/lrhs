import json
from pathlib import Path
from random import sample
from typing import List, Dict, Type, Tuple, Callable, Any
from uuid import uuid4
from functools import partial, update_wrapper
from .utils import create_hp_grid
from time import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# from http://louistiao.me/posts/adding-__name__-and-__doc__-attributes-to-functoolspartial-objects/
def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func

from .forecaster import (
    CPForecaster,
    DCTForecaster,
    DFTForecaster,
    FourierBasisRegressionForecaster,
    HoltWintersForecaster,
    SeasonalForecaster,
    TuckerForecaster, SmoothingTuckerForecaster, SmoothingCPForecaster,
)
from .utils import (
    get_param_sweep, mad, rmse, trend_cycle_decompose
)

ALL_METHODS = [method.__name__ for method in [
    CPForecaster,
    DCTForecaster,
    DFTForecaster,
    FourierBasisRegressionForecaster,
    HoltWintersForecaster,
    SeasonalForecaster,
    TuckerForecaster, SmoothingTuckerForecaster, SmoothingCPForecaster,
    ]]

class SingleForecasterExperiment:

    def __init__(
        self,
        forecaster_class: Type[SeasonalForecaster],
        param_sweep: List[int],
        folds: Tuple[int],
        error_callbacks: Tuple[Callable] = (rmse, mad),
        experiment_name: str = ""
    ):
        self.fcaster_getter = lambda x: forecaster_class(
            **x, folds=folds, error_callbacks=error_callbacks
        )
        self.param_sweep = param_sweep
        self.error_callbacks = error_callbacks
        self.forecaster_class = forecaster_class
        self.experiment_name = experiment_name

    def __call__(self, vals: pd.Series, nr_in_cycles: int, data_idx: int) -> Tuple[Dict, Dict, List, List]:

        in_errors, out_errors, results = [], [], []
        total_params = []
        for param in self.param_sweep:
            if self.forecaster_class.__name__ in ["SmoothingCPForecaster", "SmoothingTuckerForecaster"]:
                fcaster = self.fcaster_getter({**param, "data_idx": data_idx, "experiment_name": self.experiment_name})
            else:
                fcaster = self.fcaster_getter(param)

            result = fcaster(vals, nr_in_cycles=nr_in_cycles)

            results.append(result)
            in_errors.append(result.in_errors)
            out_errors.append(result.out_errors)
            total_params.append(result.nr_total_params)

        in_errors_dict, out_errors_dict = [
            dict(
                zip(
                    [f.__name__ for f in self.error_callbacks],
                    zip(*errors)
                )
            ) for errors in [in_errors, out_errors]
        ]

        return in_errors_dict, out_errors_dict, total_params, results, self.param_sweep


class TensorSeasonExperiment:
    """
    Parameters
    ----------
    dataset_name: str
        The dataset name to run the experiments on. There must be a directory under
        `datasets/` with a matching name, which contains JSON files in GluonTS data set
        format (with "start" and "target" keys).
    folds: Tuple[int]
        Number of `folds` in the multiple seasonal pattern, with the fastest index first.
        For example, (24, 7).
    nr_in_cycles: int
        Number of cycles (a cycle has np.prod(folds) length) to consider in sample.
    nr_examples: int
        `nr_examples` many time series will be sampled (without replacement) from the data set
        in order to perform experiments. If -1, all time series will be used.
    dft_sweep_length: int
        number of parameters in DFT and DCT
    tensor_sweep_length: int
        number of parameters (ranks) for tensor based methods
    n_jobs: int
        If greater than 1, joblib will be used to parallelize the experiment on `n_jobs`
        workers.
    methods: list
    """
    def __init__(
        self,
        dataset_name: str,
        folds: Tuple[int],
        nr_in_cycles: int,
        nr_test_cycles: int,
        nr_examples: int = 10,
        sarima_hps=[(1,0,1,1,0,1,24)],
        dft_sweep_length: int = 100,
        CP_Rs:List[int]=list(range(1, 8)),
        Tucker_Rs:List[int]=list(range(1, 8)),
        Tucker_time_Rs:List[int]=[1],
        l2_reg_params=[0.0],
        data_freq: str = "1h",
        n_jobs: int = 1,
        tensor_smoothing_alphas=[0.5],
        hw_alphas=[0.5],
        experiment_name="",
        dataset_path: str = "datasets/",
        methods: List[str] = ALL_METHODS,
    ) -> None:
        self.dataset_name = dataset_name
        self.nr_in_cycles = nr_in_cycles
        self.nr_test_cycles = nr_test_cycles
        self.folds = folds
        self.l2_reg_params = l2_reg_params
        self.nr_examples = nr_examples
        self.dft_sweep_length = dft_sweep_length
        self.CP_Rs = CP_Rs
        self.Tucker_Rs = Tucker_Rs
        self.data_freq = data_freq
        self.Tucker_time_Rs = Tucker_time_Rs
        self.sarima_hps = sarima_hps
        self.n_jobs = n_jobs
        self.methods = methods
        self.tensor_smoothing_alphas = tensor_smoothing_alphas
        self.hw_alphas = hw_alphas
        self.experiment_name = experiment_name
        
        data_path = Path.iterdir(Path(dataset_path) / self.dataset_name)
        self.data_path_list = [
            d for d in data_path if ".DS_Store" not in str(d)
        ]
        self.data_indices = (
            sample(range(len(self.data_path_list)), nr_examples)
            if nr_examples > 0
            else list(range(len(self.data_path_list)))
        )
        
    def _get_dataset(self) -> List[Dict]:
        dataset = []
        for i in self.data_indices:
            with open(self.data_path_list[i]) as fp:
                dataset.append(json.load(fp))        
        return dataset

    def _get_experiments(self) -> List[SingleForecasterExperiment]:
        dft_sweep = create_hp_grid({"nr_params": get_param_sweep(
            int(np.prod(self.folds)), "log", self.dft_sweep_length
        )})
        cp_sweep_smooth = create_hp_grid({
            "nr_params": self.CP_Rs,
            "l2_reg": self.l2_reg_params,
            "alpha": self.tensor_smoothing_alphas
            })
        tucker_sweep_smooth = create_hp_grid({
            "nr_params": self.Tucker_Rs,
            "time_R": self.Tucker_time_Rs,
            "alpha": self.tensor_smoothing_alphas
            })
        cp_sweep = create_hp_grid({
            "nr_params": self.CP_Rs,
            "l2_reg": self.l2_reg_params,
            "alpha": [1.0],
            })
        tucker_sweep = create_hp_grid({
            "nr_params": self.Tucker_Rs,
            "time_R": self.Tucker_time_Rs,
            "alpha": [1.0],
            })
        fbm_sweep = create_hp_grid({"nr_params": list(range(1, 40))})
        hw_sweep = create_hp_grid({"nr_params": [1], "alpha": self.hw_alphas})

        all_experiments = {
            CPForecaster.__name__: SingleForecasterExperiment(CPForecaster, cp_sweep, folds=self.folds),
            TuckerForecaster.__name__: SingleForecasterExperiment(TuckerForecaster, tucker_sweep, folds=self.folds),
            SmoothingCPForecaster.__name__: SingleForecasterExperiment(SmoothingCPForecaster, cp_sweep_smooth, folds=self.folds, experiment_name=self.experiment_name),
            SmoothingTuckerForecaster.__name__: SingleForecasterExperiment(SmoothingTuckerForecaster, tucker_sweep_smooth, folds=self.folds),
            DCTForecaster.__name__: SingleForecasterExperiment(DCTForecaster, dft_sweep, folds=self.folds),
            DFTForecaster.__name__: SingleForecasterExperiment(DFTForecaster, dft_sweep, folds=self.folds),
            FourierBasisRegressionForecaster.__name__: SingleForecasterExperiment(FourierBasisRegressionForecaster, fbm_sweep, folds=self.folds),
            HoltWintersForecaster.__name__: SingleForecasterExperiment(HoltWintersForecaster, hw_sweep, folds=self.folds),
        }

        experiments = [all_experiments[key] for key in all_experiments.keys() if key in self.methods]
        return experiments

    def run(self):
        def process_dataset(data, exp_id_, data_idx):
            frames_ = []

            time_index = pd.date_range(
                start=pd.Timestamp(data["start"]),
                periods=len(data["target"]),
                freq=self.data_freq,
            )

            orig_data_df = pd.Series(
                data["target"],
                index=time_index,
            )

            # Remove test cycles
            if self.nr_test_cycles:
                orig_data_df = orig_data_df.iloc[:-self.nr_test_cycles * np.prod(self.folds)]

            tc, vals = trend_cycle_decompose(orig_data_df, w=int(2 * np.prod(self.folds)))
            vals = vals / (vals.max() - vals.min())  # scale the residuals

            exps_ = self._get_experiments()
            for experiment in exps_:
                # try:
                    ins, outs, num_params, _, params = experiment(vals, nr_in_cycles=self.nr_in_cycles, data_idx=data_idx)

                    result_columns = {}
                    for err in ins:
                        result_columns[f"in_{err}"] = ins[err]
                    for err in outs:
                        result_columns[f"out_{err}"] = outs[err]

                    results = pd.DataFrame(
                        dict(
                            num_params=num_params,
                            **pd.DataFrame.from_dict(params, orient="columns").to_dict(orient="list"),
                            **result_columns,
                        )
                    )
                    results["experiment_id"] = exp_id_
                    results["dataset"] = self.dataset_name
                    results["model"] = experiment.forecaster_class.__name__
                    results["data_id"] = data["id"]

                    frames_.append(results)

                # except Exception as e:
                #    print(e)
                #    print(f"Exception encountered at {experiment.forecaster_class}, data id: {data['id']}")

            return pd.concat(frames_)
        dataset = self._get_dataset()
        experiment_id = str(uuid4())
        if self.n_jobs > 1:
            from joblib import Parallel, delayed
            frames = Parallel(n_jobs=self.n_jobs)(
                delayed(process_dataset)(dataset[i], experiment_id, i) for i in tqdm(range(len(dataset)))
            )
        else:
            frames = [process_dataset(data, experiment_id, i) for i, data in enumerate(tqdm(dataset))]

        return pd.concat(frames)
