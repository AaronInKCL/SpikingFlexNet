"""
- this is for hyperparameter tuning
- you can ignore this snippet
"""

from itertools import product


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def to_dict(self):
        return {key: value for key, value in self.items() if not key.startswith("__")}

    def update(self, other):
        for key, value in other.items():
            self[key] = value

    def copy(self):
        return AttrDict(self.items())


class Configurations:
    def __init__(self):
        pass

    def make_combinations(self):
        assert hasattr(self, "settings"), "You must init a setting instance first."
        keys = list(self.settings.configurations.keys())
        values = list(self.settings.configurations.values())
        self.combinations = [AttrDict(dict(zip(keys, combination))) for combination in product(*values)]

    def deduplicate(self):
        assert hasattr(self, "combinations"), "You must make combinations first."
        assert len(self.combinations) > 0, "self.combinations is empty."
        unique_dicts = []
        seen = set()

        for d in self.combinations:
            frozenset_d = frozenset(d.items())
            if frozenset_d not in seen:
                seen.add(frozenset_d)
                unique_dicts.append(d)

        self.combinations = unique_dicts

    def __getitem__(self, idx):
        return self.combinations[idx]

    def __len__(self):
        return len(self.combinations)


class Settings:
    def __init__(self, parent_configurations):
        self.parent_configurations = parent_configurations
        self.configurations = {}

    def __call__(self, name, values):
        self.configurations[name] = values
