import argparse
from sys import stdout
from progress.bar import Bar


class KeyValueAction(argparse.Action):
    """
    Custom argparse action that parses multiple key:value pairs
    into a dictionary.

    Example CLI usage:
        --model_opt encoder_dim:712 adaptor_dim:512 num_classes:2
    """

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values,
        option_string: str = None,
    ) -> None:
        model_opt_dict = {}

        for value in values:
            try:
                key, val = value.split(":")

                if val.isdigit():
                    val = int(val)
                else:
                    try:
                        val = float(val)
                    except ValueError:
                        pass

                model_opt_dict[key] = val

            except ValueError:
                raise argparse.ArgumentError(
                    self, f"Invalid key:value format: '{value}'"
                )

        setattr(namespace, self.dest, model_opt_dict)


class ProgressBar(Bar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.file = stdout
