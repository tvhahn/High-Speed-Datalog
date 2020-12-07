import abc
import json
import time
import click
from typing import Any, List, TypeVar, Type, Callable, cast
from threading import Thread, Event

from HSD.model.DeviceConfig import Device, Sensor, SensorDescriptor, SensorStatus
from HSD.communication.hsd_dll import HSD_Dll
from HSD.communication.HSDCommands import HSDCmd
from HSD.communication.STWINHSDCommands import STWINHSDCmd, STWINHSDSetSensorCmd, IsActiveParam, ODRParam, FSParam, SamplePerTSParam, MLCConfigParam
from HSD.communication.OtherDevHSDCommands import OtherDevHSDCmd

class CommandManagerFactory:
    @staticmethod
    def create_cmd_manager(dev_name: str):
        if dev_name == 'stwin':
            return STWINCommandManager()

        elif dev_name == 'otherdevice':
            return OtherDevCommandManager()

""" class HSDCmd(metaclass=abc.ABCMeta):

    @staticmethod
    @abc.abstractmethod
    def from_dict(self, obj: Any):
        pass

    @abc.abstractmethod
    def to_dict(self):
        pass """


class CommandManager(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def send_get_command(self, command: HSDCmd):
        pass

    @abc.abstractmethod
    def send_set_command(self, command: HSDCmd):
        pass

    @abc.abstractmethod
    def send_control_command(self, command: HSDCmd):
        pass
    
    @abc.abstractmethod
    def getVersion(self):
        pass

    @abc.abstractmethod
    def getNofDevices(self):
        pass

class OtherDevCommandManager(CommandManager):
    def send_get_command(self, command: OtherDevHSDCmd):
        pass

    def send_set_command(self, command: OtherDevHSDCmd):
        pass

    def send_control_command(self, command: OtherDevHSDCmd):
        pass
    
    def getVersion(self):
        return "DUMMY VERSION"

    def getNofDevices(self):
        return 7

class STWINCommandManager(CommandManager):
    
    def __init__(self):
        self.hsd_dll = HSD_Dll()
        if(not self.hsd_dll.hs_datalog_open()):
            click.secho("Error in Communication Engine opening", fg='red')
            quit()
        else:
            click.secho("Communication Engine UP (DLL)", fg='green')
    
    def __del__(self):
        if(not self.hsd_dll.hs_datalog_close()):
            click.secho("Error in Communication Engine closure", fg='red')
            quit()
        else:
            click.secho("Communication Engine DOWN (DLL)", fg='green')

    def send_get_command(self, command: STWINHSDCmd):
        pass

    def send_set_command(self, command: STWINHSDCmd):
        pass

    def send_control_command(self, command: STWINHSDCmd):
        if(self.hsd_dll):
            message = command.to_dict()
            print(message)
            self.hsd_dll.hs_datalog_send_message(0,message,len(message))
        pass
    
    def getVersion(self):
        res = self.hsd_dll.hs_datalog_get_version()
        return res[0]

    def getNofDevices(self):
        res = self.hsd_dll.hs_datalog_get_device_number()
        if(res[0]):
            if(res[1] == 0):
                return None
            else:
                return res[1]
        return None
    
    def getDeviceInfo(self, d_id: int):
        res = self.hsd_dll.hs_datalog_get_device_descriptor(d_id)
        if(res[0]):
            return res[1]
        return None

    def getDevice(self, d_id: int):
        res = self.hsd_dll.hs_datalog_get_device(d_id)
        if(res[0]):
            return res[1]
        return None

    def updateDevice(self, d_id, device_model: Device):
        sensor_list = device_model.sensor
        for sensor in sensor_list:
            ss_stat_list = sensor.sensor_status.sub_sensor_status
            for i in range(len(ss_stat_list)):
                params = []
                sss = ss_stat_list[i]
                if sss.is_active is not None:
                    params.append(IsActiveParam(i, sss.is_active))
                if sss.odr is not None:
                    params.append(ODRParam(i, sss.odr))
                if sss.fs is not None:
                    params.append(FSParam(i, sss.fs))
                if sss.samples_per_ts is not None:
                    params.append(SamplePerTSParam(i, sss.samples_per_ts))
                cmd = STWINHSDSetSensorCmd(sensor.id,params)
                cmd_dict = cmd.to_dict()
                cmd_str = json.dumps(cmd_dict)
                if not self.hsd_dll.hs_datalog_send_message(d_id,cmd_str,len(cmd_str))[0]:
                    return False
        return True
    
    def setAcquisitionParam(self, d_id: int, name: str, description: str):
        return self.hsd_dll.hs_datalog_set_acquisition_param(d_id,name,description)

    def getAcquisitionInfo(self, d_id: int):
        res = self.hsd_dll.hs_datalog_get_acquisition_info(d_id)
        if(res[0]):
            return res[1]
        return None

    def setSWTagOn(self, d_id: int, t_id: int):
        return self.hsd_dll.hs_datalog_set_on_sw_tag(d_id, t_id)
    
    def setSWTagOff(self, d_id: int, t_id: int):
        return self.hsd_dll.hs_datalog_set_off_sw_tag(d_id, t_id)

    def getAvailableTags(self, d_id: int):
        res = self.hsd_dll.hs_datalog_get_available_tags(d_id)
        if(res[0]):
            return res[1]
        return None

    def getSubSensorsFromDevice(self, d_id: int, type_filter="", only_active=True):
        res = self.getDevice(d_id)
        active_sensors = []
        if(res):
            device_dict = json.loads(res)
            device_model = Device.from_dict(device_dict['device'])
            sensor_list = device_model.sensor
            for s in sensor_list:
                active_sensor = s
                active_ss_stat_list = []
                active_ss_desc_list = []
                ss_stat_list = s.sensor_status.sub_sensor_status
                ss_desc_list = s.sensor_descriptor.sub_sensor_descriptor
                for i, sss in enumerate(ss_stat_list):
                    if type_filter == "":
                        if only_active:
                            if sss.is_active:
                                active_ss_stat_list.append(sss)
                                active_ss_desc_list.append(ss_desc_list[i])
                        else:
                            active_ss_stat_list.append(sss)
                            active_ss_desc_list.append(ss_desc_list[i])
                    else:
                        if only_active:
                            if sss.is_active and ss_desc_list[i].sensor_type == type_filter.upper():
                                active_ss_stat_list.append(sss)
                                active_ss_desc_list.append(ss_desc_list[i])
                        else:
                            if ss_desc_list[i].sensor_type == type_filter.upper():
                                active_ss_stat_list.append(sss)
                                active_ss_desc_list.append(ss_desc_list[i])
                
                active_sensor.sensor_descriptor.sub_sensor_descriptor = active_ss_desc_list
                active_sensor.sensor_status.sub_sensor_status = active_ss_stat_list
                active_sensors.append(active_sensor)
        return active_sensors    

    def getSensorData(self, d_id: int,s_id: int, ss_id: int):
        size = self.hsd_dll.hs_datalog_get_available_data_size(d_id,s_id,ss_id)
        if(size[1] > 0):
            data = self.hsd_dll.hs_datalog_get_data(d_id,s_id,ss_id,size[1])
            if data[0]:
                return [size[1], data[1]]
        return None

    def startLog(self,d_id: int):
        res = self.hsd_dll.hs_datalog_start_log(d_id)
        if res:
            click.secho("\nAcquisition started")
        else:
            print("Error in acquisition start!")
        return res
        # return None

    def stopLog(self, d_id: int):
        res = self.hsd_dll.hs_datalog_stop_log(d_id)
        if res:
            click.secho("Acquisition stopped")
        else:
            print("Acquisition stop error!")
            return False

    def updateMLCconfig(self, d_id: int, s_id: int, ucf_buffer):
        return self.hsd_dll.hs_datalog_send_UCF_to_MLC(d_id,s_id, ucf_buffer, len(ucf_buffer)) 

