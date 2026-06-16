import unittest
import json

from metaheuristics.ccea.ccea_java import app


class TestCceaInputValidation(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def _post(self, payload):
        return self.client.post(
            '/run_ccea',
            data=json.dumps(payload),
            content_type='application/json'
        )

    def test_string_keys_normalized_and_runs(self):
        payload = {
            "config_seru": {
                "num_of_workers": 2,
                "num_of_batches": 2,
                "max_num_of_multiple_task": 0,
                "setup_time": 0,
                "use_standard_logic": False
            },
            "problem_data": {
                "worker_to_task_dict": {
                    "1": {"系数": 0.5},
                    "2": {"系数": 0.5}
                },
                "worker_to_product_dict": {
                    "1": {"1": 1.0},
                    "2": {"1": 1.0}
                },
                "batch_to_product_dict": {
                    "1": {"产品类型": 1, "批次大小": 10},
                    "2": {"产品类型": 1, "批次大小": 10}
                },
                "batch_due_dates_dict": {
                    "1": {"批次截止时间": 100.0},
                    "2": {"批次截止时间": 100.0}
                }
            }
        }

        resp = self._post(payload)
        self.assertEqual(resp.status_code, 200, msg=resp.data)

    def test_missing_batch_ids_returns_400(self):
        payload = {
            "config_seru": {
                "num_of_workers": 2,
                "num_of_batches": 3
            },
            "problem_data": {
                "worker_to_task_dict": {
                    1: {"系数": 0.5},
                    2: {"系数": 0.5}
                },
                "worker_to_product_dict": {
                    1: {1: 1.0},
                    2: {1: 1.0}
                },
                "batch_to_product_dict": {
                    1: {"产品类型": 1, "批次大小": 10},
                    2: {"产品类型": 1, "批次大小": 10}
                },
                "batch_due_dates_dict": {
                    1: {"批次截止时间": 100.0},
                    2: {"批次截止时间": 100.0}
                }
            }
        }

        resp = self._post(payload)
        self.assertEqual(resp.status_code, 400, msg=resp.data)

    def test_missing_setup_time_defaults_to_zero(self):
        payload = {
            "config_seru": {
                "num_of_workers": 2,
                "num_of_batches": 2,
                "max_num_of_multiple_task": 0,
                "use_standard_logic": False
            },
            "problem_data": {
                "worker_to_task_dict": {
                    1: {"系数": 0.5},
                    2: {"系数": 0.5}
                },
                "worker_to_product_dict": {
                    1: {1: 1.0},
                    2: {1: 1.0}
                },
                "batch_to_product_dict": {
                    1: {"产品类型": 1, "批次大小": 10},
                    2: {"产品类型": 1, "批次大小": 10}
                },
                "batch_due_dates_dict": {
                    1: {"批次截止时间": 100.0},
                    2: {"批次截止时间": 100.0}
                }
            }
        }
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 200, msg=resp.data)

    def test_missing_due_dates_no_error(self):
        payload = {
            "config_seru": {
                "num_of_workers": 2,
                "num_of_batches": 2,
                "max_num_of_multiple_task": 0,
                "use_standard_logic": False
            },
            "problem_data": {
                "worker_to_task_dict": {
                    1: {"系数": 0.5},
                    2: {"系数": 0.5}
                },
                "worker_to_product_dict": {
                    1: {1: 1.0},
                    2: {1: 1.0}
                },
                "batch_to_product_dict": {
                    1: {"产品类型": 1, "批次大小": 10},
                    2: {"产品类型": 1, "批次大小": 10}
                }
            }
        }
        resp = self._post(payload)
        self.assertEqual(resp.status_code, 200, msg=resp.data)


if __name__ == '__main__':
    unittest.main()