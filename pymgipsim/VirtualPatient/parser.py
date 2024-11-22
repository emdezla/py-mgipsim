import argparse
from ..Utilities.paths import default_settings_path, models_path
from ..Utilities.Scenario import load_scenario
from pymgipsim.VirtualPatient.Models import T1DM
import glob
from pymgipsim.Interface.Messages.parser_colors import color_help_text, color_group_header_text

default_settings = load_scenario(default_settings_path + "\\scenario_default.json")

def get_model_names():
    paths = glob.glob(models_path + "\\*\\*\\")
    models = [folder for folder in paths if 'T1DM' in folder and '__' not in folder]
    models = [folder.split("\\")[-3:] for folder in models]
    models = [folder[0] + "." + folder[1] for folder in models]
    return models

def generate_virtual_subjects_parser( parent_parser = [], add_help = True):

    parser = argparse.ArgumentParser(
                                    prog = 'Virtual Subject Settings',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                    parents=parent_parser,
                                    add_help=add_help
                                    )

    ######################################################################################

    model_group = parser.add_argument_group(color_group_header_text('Model and Diabetes Type'))

    # match mode:
    #
    #     case 'singlescale':
    #         model_names = get_model_names()
    #
    #     case 'multiscale':
    #         model_names = [T2DM.Jauslin.Model.name]

    model_group.add_argument(
                        '-mn',
                        '--model-name',
                        help = color_help_text('Name of the model that will be used.'),
                        dest = 'model_name',
                        default = T1DM.ExtHovorka.Model.name,
                        choices = get_model_names(),
                        )

    model_group.add_argument(
                        '-pn',
                        '--patient-names',
                        help = color_help_text('Names of specific patient file(s) to be used instead of the whole population if desired. By default, all patients are simulated. ' + \
                                                'Any number of subjects <= 20 can be used.'),
                        dest = 'patient_names',
                        nargs='*'
                        )
    

    subject_traits_group = parser.add_argument_group(title = color_group_header_text('Subject and Traits'))


    subject_traits_group.add_argument(
                                    '-rf',
                                    '--renal_function',
                                    help=color_help_text('Category of renal function (1-5). Default is 1 (normal function) while 5 is the lowest (kidney failure).'),
                                    dest='renal_function_category',
                                    type=int,
                                    default = default_settings.patient.demographic_info.renal_function_category
                                    )
    
    subject_traits_group.add_argument(
                                    '-bwr',
                                    '--body-weight-range',
                                    help=color_help_text('Range of body weight (in kg) to sample from for virtual subjects.'),
                                    dest='body_weight_range',
                                    nargs=2,
                                    type = int,
                                    default = default_settings.patient.demographic_info.body_weight_range
                                    )

    subject_traits_group.add_argument(
                                    '-ns',
                                    '--number_of_subjects',
                                    help=color_help_text('Number of subjects in simulation. This should be <= 20.'),
                                    dest='number_of_subjects',
                                    default=default_settings.patient.number_of_subjects,
                                    type=int
                                    )

    return parser


def generate_results_parser(parent_parser = [], add_help = True):

    parser = argparse.ArgumentParser(
                                prog = 'Simulation Results Parser',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                parents=parent_parser,
                                add_help=add_help
                                )

    output_formatting_group = parser.add_argument_group(color_group_header_text('Format Results'))

    output_formatting_group.add_argument('-xl',
                                        '--to-excel',
                                        help = color_help_text('Save the model results in xlsx format. WARNING: Significantly increases run time'),
                                        action = 'store_true',
                                        dest = 'to_excel'
                                        )

    return parser
