import os
import sys
import click
from .file_manager import FileManager
from HSD.model.DeviceConfig import Device, Sensor
from HSD.model.AcquisitionInfo import Tag
class CLIInteraction:

    @staticmethod
    def select_datalog_folder(dev_config_name: str, acq_info_name: str, path=None):
        """prompt user to select Datalog folder"""
        if path is None:
            cwd = os.getcwd()
        else:
            cwd = path
        files = FileManager.hsd_get_list_of_sub_folders(dev_config_name, acq_info_name, cwd)
        base_names = [b[1] for b in files]
        s = CLIInteraction.select_item('Folder', base_names)
        return [b[0] for b in files if b[1] == s][0]

    @staticmethod
    def select_item(what: str, list: list):
        choice_not_in_range = True
        choice = 0
        item_id = ""
        while choice_not_in_range:
            for i, c in enumerate(list):
                if isinstance(c , Device):
                    item_id = c.device_info.alias + " [" + c.device_info.part_number + "]"
                elif isinstance(c , Sensor):
                    item_id = c.name
                elif isinstance(c, Tag):
                    item_id = c.label
                else:
                    item_id = str(c)
                click.secho(str(i) + ' - ' + item_id)

            if len(list) == 0:
                click.secho('==> No {w} in list'.format(w=what), fg='red')
                return None

            print('q - Quit')
            choice = input('Select one {} (''q'' to quit) ==> '.format(what))
            try:
                choice_not_in_range = int(choice) not in range(len(list))
                if choice_not_in_range:
                    click.secho('please input a correct value',fg='red')
            except ValueError:
                if choice != 'q':
                    click.secho('please input a correct value',fg='red')
                else:
                    #choice_not_in_range = False
                    break
        if choice == 'q':
            click.echo('Bye!')
            return None
        else:
            return list[int(choice)]