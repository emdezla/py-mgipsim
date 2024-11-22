import argparse, json, pprint
from ..Utilities.paths import default_settings_path
from ..Utilities.Scenario import scenario, load_scenario
import colorama
from pymgipsim.Interface.Messages.parser_colors import color_help_text, color_group_header_text

# with open(default_settings_path + "\\scenario_default.json","r") as f:
#     default_settings = scenario(**json.load(f))
# f.close()

default_settings = load_scenario(default_settings_path + "\\scenario_default.json")

def generate_load_parser(parent_parser = [], add_help = True):

    parser = argparse.ArgumentParser(
                                    prog = 'Load Scenario',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    parents = parent_parser,
                                    add_help = add_help
                                    )

    load_scenario_group = parser.add_argument_group(color_group_header_text('Load Scenario'))
    load_scenario_group.add_argument(
                                    '-sn',
                                    '--scenario-name',\
                                    help = color_help_text("Scenario file name"),\
                                    dest = 'scenario_name',\
                                    type = str
                                    )

    return parser