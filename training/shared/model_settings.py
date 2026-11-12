import itertools
import copy

# define settings
def extend_settings(settings, addSettings):
    extended_settings = []

    # Get all keys and value lists from addSettings
    add_keys = list(addSettings.keys())
    add_values_lists = [addSettings[key] for key in add_keys]

    # Generate all combinations of additional settings
    combinations = list(itertools.product(*add_values_lists))

    for base_setting in settings:
        for combo in combinations:
            # Deep copy to avoid modifying the original
            new_setting = copy.deepcopy(base_setting)
            for key, value in zip(add_keys, combo):
                new_setting[key] = value
            extended_settings.append(new_setting)

    return extended_settings
