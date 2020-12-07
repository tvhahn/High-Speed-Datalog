import os
import json
import ntpath
import numpy as np


class JsonControlUtils:
    @staticmethod
    def get_active_sensors_from_json(cj):
        active_dev = [t['sensorDescriptor']['name'] for t in cj['device']['sensor'] if t['sensorStatus']['isActive']]
        return active_dev

    #TODO mmmmmm
    @staticmethod
    def get_files_from_json(cj):
        """
        list of files that are supposed to be in the datalog based on DeviceConfig.json
        """
        data_files_list = []
        for d in cj['device']['sensor']:
            sensor_name = d['name']
            for idx, s in enumerate(d['sensorStatus']['subSensorStatus']):
                if s['isActive']:
                    sub_name = d['sensorDescriptor']['subSensorDescriptor'][idx]['sensorType']
                    file_name = sensor_name + '_' + sub_name + '.dat'
                    data_files_list.append(file_name)
        return data_files_list

class FileManager:

    @staticmethod
    def decode_file_name(file_name):
        [sensor_name, sub] = file_name.split(".")[0].split("_")
        return sensor_name, sub

    @staticmethod
    def encode_file_name(sensor_name, sub_sensor_type, ext = '.dat'):
        file_name = sensor_name + '_' + sub_sensor_type + ext
        return file_name

    @staticmethod
    def get_json_file(fp, jf_name="DeviceConfig.json"):
        """Opens a JSON file ( by default the Device Config ) from path
        """
        cfn = os.path.join(fp, jf_name)
        with open(cfn) as jf:
            cj = json.load(jf)
        return cj

    @staticmethod
    def get_files_from_model(device):
        """
        list of files that are supposed to be in the datalog based on DeviceConfig.json
        """
        data_files_list = []
        for s in device.sensor:
            sss_list = s.sensor_status.sub_sensor_status
            for i, ssd in enumerate(s.sensor_descriptor.sub_sensor_descriptor):
                if sss_list[i].is_active:
                    file_name = FileManager.encode_file_name(s.name,ssd.sensor_type)
                    data_files_list.append(file_name)
        return data_files_list

    @staticmethod
    def get_files_from_json(cj):
        JsonControlUtils.get_files_from_json(cj)

    @staticmethod
    def hsd_get_list_of_sub_folders(dev_config_json_file_name, acq_info_json_file_name, path=None):
        """
        hsd_get_list_of_sub_folders(dev_config_json_file_name, acq_info_json_file_name, path=None)
        dev_config_json_file_name: TODO
        acq_info_json_file_name: TODO
        path: optional parameter. default is current path (as from os.getcwd()
        returns an np.array [full path, basename]
        full path = full path of folders with a valid Datalog
        basename = basename of folders (useful foe GUIs)
        """
        if path is None:
            cwd = os.getcwd()
        else:
            cwd = path
        folders = next(os.walk(cwd))[1]
        fps = [os.path.join(cwd, f) for f in folders]
        fps_ok = []

        for fp in fps:
            if FileManager.hsd_check_folder_structure(fp, dev_config_json_file_name, acq_info_json_file_name, verbose=False)[0]:
                fps_ok = np.append(fps_ok, fp)
        ret = [[pa, ntpath.basename(pa)] for pa in fps_ok]
        return ret

    @staticmethod
    def hsd_check_folder_structure(fp, dev_config_json_file_name, acq_info_json_file_name,  verbose=True):
        """
        returns [isConfigured, isLabelled, dataFilesExistReport]
        isConfigured is a boolean => folder and Config JSON file exist
        isLabelled is a boolean => labels JSON file exist in folder
        dataFilesExistReport is a list of tuples (filename, boolean => data file exist in folder)
        """
        is_configured = False
        is_labelled = False
        data_files_exist_report = [False]

        if os.path.exists(fp) and os.path.exists(dev_config_json_file_name):
            cj = FileManager.get_json_file(fp, jf_name=dev_config_json_file_name)
            data_files = JsonControlUtils.get_files_from_json(cj)
            data_files_path = [os.path.join(fp, f) for f in data_files]
            data_files_exist = [os.path.exists(f) for f in data_files_path]
            data_files_exist_report = [(f, os.path.exists(os.path.join(fp, f))) for f in data_files]

            is_configured = all(data_files_exist)
            if os.path.exists(acq_info_json_file_name):
                is_labelled = True
        elif verbose:
            print('Failed existence check in datalog folder:')
            print('path: ', os.path.exists(fp))
            print(acq_info_json_file_name, 'file:', os.path.exists(acq_info_json_file_name))
            print(dev_config_json_file_name, 'file:', os.path.exists(dev_config_json_file_name))
        return [is_configured, is_labelled, data_files_exist_report]
