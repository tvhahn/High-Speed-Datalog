import abc
import json
import os
import sys
import click
import math
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from HSD.communication.commands import CommandManagerFactory, CommandManager
from HSD.model.DeviceConfig import Device, DeviceInfo, Sensor, SensorStatus, SensorDescriptor, SubSensorStatus, SubSensorDescriptor, TagConfig
from HSD.model.AcquisitionInfo import AcquisitionInfo, Tag
from HSD.utils.cli_interaction import CLIInteraction as CLI
from HSD.utils.file_manager import FileManager as FM

class HSDatalog:
    __command_manager: CommandManager = None
    device_model = []
    acq_info_model: AcquisitionInfo
    __acq_folder_path = None
    __start_acq_time: float
    __tagList = []
    __checkTimestamps = False
    
    def __init__(self, acquisition_folder = None):
        self.__acq_folder_path = acquisition_folder
        if acquisition_folder is not None:
            self.__loadDevice(acquisition_folder)
            self.__loadAcquisitionInfo(acquisition_folder)
        self.__acq_time = 0.0
        self.__tagList = []
        self.__tagSWClassesStatus = []
        self.__tagHWClassesStatus = []
    
    #========================================================================================#
    ### ONLINE Device/Log Control ############################################################
    #========================================================================================#
    
    def initDeviceCommandSet(self, command_set: str):
        self.__command_manager = CommandManagerFactory.create_cmd_manager(command_set)

    ### DEVICE INFO Commands ###
    def getVersion(self):
        return self.__command_manager.getVersion()

    def getNofDevices(self):
        return self.__command_manager.getNofDevices()

    def getDeviceInfo(self, d_id: int):
        res = self.__command_manager.getDeviceInfo(d_id)
        if res is not None:
            dev_info_dict = json.loads(res)
            return DeviceInfo.from_dict(dev_info_dict['deviceInfo'])
        print("No device info returned.")
        return None

    def __getDevice(self, d_id: int):
        res = self.__command_manager.getDevice(d_id)
        if res is not None:
            device_dict = json.loads(res)
            res_dev = Device.from_dict(device_dict['device'])
            return res_dev
        print("No device returned.")
        return None

    def saveJSONDevicefromRealDevice(self, d_id: int, out_acq_path):
        if not os.path.exists(out_acq_path):
            os.makedirs(out_acq_path)
        res = self.__command_manager.getDevice(d_id)
        if res is not None:
            device_config_filename = os.path.join(out_acq_path,"DeviceConfig.json")
            sensor_data_file = open(device_config_filename, "w+")
            sensor_data_file.write(res)
            sensor_data_file.close()
            return True
        else:
            return False
    
    def getConnectedDevices(self):
        nof_devices = self.getNofDevices()
        if nof_devices is not None:
            dev_list=[]
            for i in range(0,nof_devices):
                device = self.__getDevice(i)
                dev_list.append(device)
            self.device_model = dev_list
            return dev_list
        return None

    ### DEVICE CONFIG Commands ###
    def updateDevice(self, d_id: int, device_json_file_folder):
        self.__loadDevice(device_json_file_folder, d_id)
        return self.__command_manager.updateDevice(d_id, self.device_model[d_id])
    
    def updateMLCconfig(self, d_id, s_id, ucf_file):
        with open(ucf_file, 'rb') as fid:
            ucf_buffer = np.fromfile(fid, dtype='uint8')
        return self.__command_manager.updateMLCconfig(d_id, s_id, ucf_buffer)
    
    ### TAGs CONFIG COMMANDS ###
    def getAvailableTags(self, d_id: int):
        res = self.__command_manager.getAvailableTags(d_id)
        if (res):
            tags_dict = json.loads(res)
            tags_model = TagConfig.from_dict(tags_dict)
            return tags_model
        print("No available tags.")
        return None
    
    def getSWTagClasses(self, d_id: int):
        tagConfig = self.getAvailableTags(d_id)
        if (tagConfig):
            return tagConfig.sw_tags
    
    def getHWTagsClasses(self, d_id: int):
        tagConfig = self.getAvailableTags(d_id)
        if (tagConfig):
            return tagConfig.hw_tags
    
    def getMaxTagsPerAcq(self, d_id: int):
        tagConfig = self.getAvailableTags(d_id)
        if (tagConfig):
            return tagConfig.max_tags_per_acq

    ### ACQUISITION INFO Commands ###
    ### ==> utils ###
    def __getAcqInfoFromSTR(self, acq_info_str: str):
        acq_info_dict = json.loads(acq_info_str)
        return AcquisitionInfo.from_dict(acq_info_dict)
    ### utils <== ###

    def getAcquisitionInfo(self, d_id: int):
        res = self.__command_manager.getAcquisitionInfo(d_id)
        if(res):
            return self.__getAcqInfoFromSTR(res)
        print("No acquisition info returned.")
        return None

    def setAcquisitionParam(self, d_id: int, name, description):
        return self.__command_manager.setAcquisitionParam(d_id,name,description)

    def saveJSONAcqInfofromRealDevice(self, d_id: int, out_acq_path):
        if not os.path.exists(out_acq_path):
            os.makedirs(out_acq_path)
        res = self.__command_manager.getAcquisitionInfo(d_id)
        if res is not None:
            acq_info_filename = os.path.join(out_acq_path,"AcquisitionInfo.json")
            acq_info_file = open(acq_info_filename, "w+")
            acq_info_file.write(res)
            acq_info_file.close()

    ### ACQUISITION CONTROL Commands ###
    def set_sw_tag(self, d_id: int, t_id: int, status: bool):
        if status:
            return self.__command_manager.setSWTagOn(d_id,t_id)
        return self.__command_manager.setSWTagOff(d_id,t_id)

    def get_sub_sensors(self, d_id: int):
        return self.__command_manager.getSubSensorsFromDevice(d_id, only_active=False)

    def get_active_sub_sensors(self, d_id: int):
        return self.__command_manager.getSubSensorsFromDevice(d_id, only_active=True)

    def get_active_mlc_sensors(self, d_id: int):
        return self.__command_manager.getSubSensorsFromDevice(d_id,type_filter="MLC")

    def read_sensor_data(self, d_id: int, s_id: int, ss_id: int):
        return self.__command_manager.getSensorData(d_id,s_id,ss_id)

    def startLog(self, d_id: int):
        self.__start_acq_time = datetime.now()
        return self.__command_manager.startLog(d_id)

    def stopLog(self,d_id: int):
        return self.__command_manager.stopLog(d_id)
    
    #========================================================================================#
    ### OFFLINE Data Analisys ################################################################
    #========================================================================================#

    ### ==> utils ###
    def checkTypeLength(self, check_type):
        switcher = {
        'uint8_t': 1,
        'int8_t': 1,
        'uint16_t': 2,
        'int16_t': 2,
        'uint32_t': 4,
        'int32_t': 4,
        'float': 4,
        'double': 8,
        }
        return switcher.get(check_type, "error")

    def checkType(self, check_type):
        switcher = {
            'uint8_t': 'uint8',
            'uint16_t': 'uint16',
            'uint32_t': 'uint32',
            'int8_t': 'int8',
            'int16_t': 'int16',
            'int32_t': 'int32',
            'float': 'float32',
            'double': 'double',
        }
        return switcher.get(check_type, "error")
    
    def encodeFileName(self, sensor_name, sub_sensor_type, ext=''):
        return FM.encode_file_name(sensor_name, sub_sensor_type, ext)

    def decodeFileName(self, file_name):
        return FM.decode_file_name(file_name)
    ### utils <== ###

    ### ==> Debug ###
    def enableTimestampRecovery(self, status):
        self.__checkTimestamps = status
    ### Debug <== ###

    def __loadDevice(self, acq_folder_path, d_id=0):
        self.__acq_folder_path = acq_folder_path
        device_json_file_path = os.path.join(acq_folder_path,"DeviceConfig.json")
        with open(device_json_file_path) as f:
            device_json_dict = json.load(f)
        device_json_str = json.dumps(device_json_dict)
        f.close()
        device_dict = json.loads(device_json_str)
        if self.__command_manager is not None and len(self.device_model) != 0:
            self.device_model[d_id] = Device.from_dict(device_dict['device'])
        else:
            self.device_model.append(Device.from_dict(device_dict['device']))

    def getFileList(self):
        if self.device_model[0] is not None:
            return FM.get_files_from_model(self.device_model[0])
        print("Error! Empty Device model.")
        return None

    def __loadAcquisitionInfo(self, acq_folder_path):
        self.__acq_folder_path = acq_folder_path
        acq_info_json_file_path = os.path.join(acq_folder_path,"AcquisitionInfo.json")
        with open(acq_info_json_file_path) as f:
            acq_info_json_dict = json.load(f)
        acq_info_json_str = json.dumps(acq_info_json_dict)
        f.close()
        acq_info_dict = json.loads(acq_info_json_str)
        self.acq_info_model = AcquisitionInfo.from_dict(acq_info_dict)

    def getAcquisitionLabelClasses(self):
        if self.acq_info_model is not None:
            return sorted(set(dic.label for dic in self.acq_info_model.tags))
        print("Error! Empty Acquisition Info model.")
        return None

    def getTimeTags(self):
        # for each label and for each time segment:
        # time_labels: array of tag
        #   = {'label': lbl, 'time_start': t_start, 'time_end': xxx, }
        time_labels = []
        tags = self.acq_info_model.tags
        for lbl in self.getAcquisitionLabelClasses():

            # start_time, end_time are vectors with the corresponding 't' entries in DataTag-json
            start_time = np.array([t.t for t in tags if t.label == lbl and t.enable])
            end_time = np.array([t.t for t in tags if t.label == lbl and not t.enable])

            # now must associate at each start tag the appropriate end tag
            # (some may be missing because of errors in the tagging process)
            for tstart in start_time:
                tag = {}
                jj = [i for (i, n) in enumerate(end_time) if n >= tstart]
                if jj:
                    tend = end_time[min(jj)]
                else:
                    tend = float(-1)  # if no 'end tag' found the end is eof
                tag['Label'] = lbl
                tag['time_start'] = tstart
                tag['time_end'] = tend
                time_labels.append(tag)
        return time_labels

    def get_dataStreamTags(self, sensor_name, sub_sensor_type, sample_start = 0, sample_end = -1):
        """
        returns an array of dict:
        {'Label': <Label>, 'time_start': <time start: float>, 'time_end': <time end: float>,
                                'sample_start': <sample index start: int>, 'sample_end': <sample index end: int>}
        """
        res = self.read_datalog(sensor_name, sub_sensor_type, sample_start, sample_end)
        if res is not None:
            stream_time = res[1]
            st_tim = np.reshape(stream_time, -1)
            ind_sel = np.array(range(len(st_tim)))
            sensor_labels = []

            for tag in self.getTimeTags():
                sampleTag = {}
                tend = float(tag['time_end'])
                tend = tend if tend >= 0 else st_tim[-1]
                tstart = float(tag['time_start'])

                ind_inf = st_tim <= tend
                ind_sup = st_tim >= tstart
                ind_both = np.logical_and(ind_inf, ind_sup)
                jj = ind_sel[ind_both]

                if len(jj) > 0:
                    s_start = min(jj)
                    s_end = max(jj)
                    sampleTag = {'Label': tag["Label"], 'time_start': tag['time_start'], 'time_end': tag['time_end'],
                                    'sample_start': s_start, 'sample_end': s_end}
                else:
                    click.secho("Warning! No data samples corresponding to time tag [{}] were found in selected sample interval".format(tag['Label']), fg="yellow")

                sensor_labels.append(sampleTag)

            return sensor_labels 
            
        return None

    def getSensor(self, sensor_name, d_id = 0):
        for s in self.device_model[d_id].sensor:
            if s.name == sensor_name:
                return s
        return None

    def getDataFileList(self):
        file_names = []
        with os.scandir(self.__acq_folder_path) as listOfEntries:
            for entry in listOfEntries:
                # print all entries that are files
                if entry.is_file():
                    if entry.name.endswith('.dat'):
                        file_names.append(entry.name)
        return file_names

    def getSensorList(self, d_id = 0, only_active = False):
        if only_active:
            active_sensors = []
            for s in self.device_model[d_id].sensor:
                new_s = s
                res = self.getSubSensors(s.name)
                if res is not None:
                    ss_desc_list, ss_stat_list = res
                    new_sss_list = []
                    new_ssd_list = []
                    for i, sss in enumerate(ss_stat_list):
                        if sss.is_active:
                            new_sss_list.append(sss)
                            new_ssd_list.append(ss_desc_list[i])
                    new_s.sensor_status.sub_sensor_status = new_sss_list
                    new_s.sensor_descriptor.sub_sensor_descriptor = new_ssd_list
                    active_sensors.append(new_s)
            return active_sensors
        else:
            return self.device_model[d_id].sensor

    def isMLCSensorActive(self, sensor_name):
        return self.getSubSensor(sensor_name,ss_type='MLC')[1].is_active

    def getSubSensors(self, sensor_name, only_active = False):
        sensor = self.getSensor(sensor_name)
        if sensor is not None:
            ss_desc_list = sensor.sensor_descriptor.sub_sensor_descriptor
            ss_stat_list = sensor.sensor_status.sub_sensor_status
            if only_active:
                for i, sss in enumerate(ss_stat_list):
                    if not sss.is_active:
                        ss_desc_list.pop[i]
                        ss_stat_list.pop[i]
            return [ss_desc_list, ss_stat_list]
        return None

    def getSubSensor(self, sensor_name, ss_id = None, ss_type = None):
        sensor = self.getSensor(sensor_name)
        if sensor is not None:
            ss_desc_list = sensor.sensor_descriptor.sub_sensor_descriptor
            ss_stat_list = sensor.sensor_status.sub_sensor_status
            if ss_id is not None:
                return [ss_desc_list[ss_id], ss_stat_list[ss_id]]
            elif ss_type is not None:
                for i in range(len(ss_desc_list)):
                    if ss_desc_list[i].sensor_type == ss_type:
                        return [ss_desc_list[i], ss_stat_list[i]]
        return None        

    def __process_datalog(self, sensor_name, ss_desc, ss_stat, raw_data, dataframe_size, timestamp_size, raw_flag = False):

        #####################################################################
        def get_data_and_timestamps():
        
            """ gets data from a file .dat
                np array with one column for each axis of each active subSensor
                np array with sample times
            """

            if ss_desc.sensor_type != 'MLC':
                checkTimeStamps = self.__checkTimestamps
                frame_period = ss_stat.samples_per_ts / ss_stat.odr
            else:
                checkTimeStamps = False
                frame_period = 0

            # rndDataBuffer = raw_data rounded to an integer # of frames
            rndDataBuffer = raw_data[0:int(frame_size * num_frames)]

            data = np.zeros((data1D_per_frame * num_frames, 1), dtype = self.checkType(ss_desc.data_type))
            
            timestamp_first = ss_stat.initial_offset
            timestamp = []

            for ii in range(num_frames):  # For each Frame:
                startFrame = ii * frame_size
                # segmentData = data in the current frame
                segmentData = rndDataBuffer[startFrame: startFrame + dataframe_size]
                # segmentTS = ts is at the end of each frame
                segmentTS = rndDataBuffer[startFrame + dataframe_size: startFrame + frame_size]

                # timestamp of current frame
                if segmentTS.size != 0:
                    timestamp.append(np.frombuffer(segmentTS, dtype = 'double'))
                else:
                    timestamp_first += frame_period
                    timestamp.append(timestamp_first)

                # Data of current frame
                data[ii * data1D_per_frame:(ii + 1) * data1D_per_frame, 0] = \
                    np.frombuffer(segmentData, dtype = self.checkType(ss_desc.data_type))

                # Check Timestamp consistency
                if checkTimeStamps and (ii > 0):
                    deltaTS = timestamp[ii] - timestamp[ii - 1]
                    if abs(deltaTS) < 0.66 * frame_period or abs(deltaTS) > 1.33 * frame_period or np.isnan(
                            timestamp[ii]) or np.isnan(timestamp[ii - 1]):
                        data[ii * data1D_per_frame:(ii + 1) * data1D_per_frame, 0] = 0
                        timestamp[ii] = timestamp[ii - 1] + frame_period
                        print('WARNING Sensor {} - {}: corrupted data at'.format(sensor_name, ss_desc.sensor_type),
                                timestamp[ii], 'sec')

            timestamp = np.append(ss_stat.initial_offset, timestamp)

            # when you have only 1 frame, framePeriod = last timestamp
            if num_frames == 1:
                timestamp = np.append(timestamp, frame_period)

            sData = np.reshape(data, (-1, ss_desc.dimensions)).astype(dtype=float)

            if not raw_flag:
                sensitivity = float(ss_stat.sensitivity)
                for kk in range(ss_desc.dimensions):
                    sData[:, int(kk)] = sData[:, int(kk)] * sensitivity

            # samples_time: numpy array of 1 clock value per each data sample
            samples_time = np.zeros((num_frames * ss_stat.samples_per_ts, 1))
            # sample times between timestamps are linearly interpolated
            for ii in range(num_frames):  # For each Frame:
                temp = np.linspace(timestamp[ii], timestamp[ii + 1], ss_stat.samples_per_ts, endpoint=False)
                samples_time[ii * ss_stat.samples_per_ts:(ii + 1) * ss_stat.samples_per_ts, 0] = temp

            return sData, samples_time
        #####################################################################
        
        # size of the frame. A frame is data + ts
        frame_size = dataframe_size + timestamp_size

        # number of frames = round down (//) len datalog // frame_size
        num_frames = len(raw_data) // frame_size

        # data1D_per_frame = number of data samples in 1 frame
        # must be the same as samplePerTs * number of axes
        data1D_per_frame = int(dataframe_size / self.checkTypeLength(ss_desc.data_type))

        if ss_stat.samples_per_ts == 0:
            ss_stat.samples_per_ts = int(data1D_per_frame / ss_desc.dimensions)

        return get_data_and_timestamps()
    
    def read_datalog(self, sensor_name, sub_sensor_type, sample_start = 0, sample_end = -1, raw_flag = False):

        # get sub sensor descriptor and status
        ss_desc, ss_stat = self.getSubSensor(sensor_name, ss_type=sub_sensor_type)

        # get dat file path (obtained from "sensor_name + sub_sensor_type")
        file_path = os.path.join(self.__acq_folder_path, FM.encode_file_name(sensor_name,ss_desc.sensor_type))
        if not os.path.exists(file_path):
            error_msg = ("No such file or directory: {path}".format(path=file_path))
            click.secho(error_msg, fg='red')
            return None

        try:
            # Sample per Ts == 0 #######################################################################           
            if ss_stat.samples_per_ts == 0:
                dataframe_byte_size = int(ss_desc.dimensions * self.checkTypeLength(ss_desc.data_type))
                timestamp_byte_size = 0

                n_of_samples = sample_end - sample_start

                blocks_before_ss = 0

                if sample_end == -1:
                    n_of_samples = int((os.path.getsize(file_path) - os.path.getsize(file_path) % (self.checkTypeLength(ss_desc.data_type) * ss_desc.dimensions)) / self.checkTypeLength(ss_desc.data_type))
                    sample_end = n_of_samples
                
                read_start_bytes = sample_start * (self.checkTypeLength(ss_desc.data_type) * ss_desc.dimensions)
                read_end_bytes = sample_end * (self.checkTypeLength(ss_desc.data_type) * ss_desc.dimensions)

            # Sample per Ts != 0 #######################################################################
            else:
                dataframe_byte_size = int(ss_stat.samples_per_ts * ss_desc.dimensions * self.checkTypeLength(ss_desc.data_type))
                timestamp_byte_size = self.checkTypeLength('double')

                if sample_end == -1:
                    n_of_blocks_in_file = math.floor(os.path.getsize(file_path)/(timestamp_byte_size + dataframe_byte_size))
                    sample_end = n_of_blocks_in_file * ss_stat.samples_per_ts

                n_of_samples = sample_end - sample_start
                
                blocks_before_ss = math.floor(sample_start/(ss_stat.samples_per_ts))
                blocks_before_se = math.floor(sample_end/(ss_stat.samples_per_ts))

                if blocks_before_ss == 0:
                    read_start_bytes = 0
                else:
                    read_start_bytes = (blocks_before_ss * dataframe_byte_size) + ((blocks_before_ss - 1) * timestamp_byte_size)
                read_end_bytes = ((blocks_before_se + 1) * dataframe_byte_size) + ((blocks_before_se + 1) * timestamp_byte_size)

            with open(file_path, "rb") as f:
                f.seek(read_start_bytes)
                f_data = f.read(read_end_bytes - read_start_bytes)
                if not f_data:
                    return None
                raw_data = np.fromstring(f_data, dtype='uint8')

                real_start = sample_start + blocks_before_ss
                timestamp_pre_id = max(0, blocks_before_ss * ss_stat.samples_per_ts + blocks_before_ss - 1)

                if blocks_before_ss == 0:
                    data_offset_pre = real_start - timestamp_pre_id
                else:
                    data_offset_pre = real_start - timestamp_pre_id - 1

                # if the start_sample isn't in the first block (pre_t_bytes_id != 0)
                if read_start_bytes != 0 :
                    first_timestamp = raw_data[:timestamp_byte_size]
                    ss_stat.initial_offset = np.frombuffer(first_timestamp, dtype = 'double')
                    #remove the first timestamp
                    raw_data = raw_data[timestamp_byte_size:] 

                data, timestamp = self.__process_datalog(sensor_name, ss_desc, ss_stat, raw_data, \
                                                            dataframe_byte_size, timestamp_byte_size,
                                                            raw_flag = raw_flag )

                #trim results to obtain only the requested [data,timestamp]
                data = data[data_offset_pre:]
                data = data[:n_of_samples]
                timestamp = timestamp[data_offset_pre:]
                timestamp = timestamp[:n_of_samples]

                #DEBUG
                # click.secho("data Len: {}".format(len(data)),fg='yellow')
                # click.secho("Time Len: {}".format(len(timestamp)),fg='yellow')

            return data, timestamp

        except MemoryError:
            click.secho("Memory Error occoured! You should batch process your {} file".format(file_path), fg='red')
            sys.exit(0)
        except OverflowError:
            click.secho("Memory Error occoured! You should batch process your {} file".format(file_path), fg='red')
            sys.exit(0)

    def __to_dataFrame(self, data, time, ss_desc, sensor_name, sub_sensor_type, labeled = False, sample_start = 0, sample_end = -1):
        if data is not None and time is not None:
            cols = []
            numAxes = int(ss_desc.dimensions)
            if not ss_desc.sensor_type == "MLC":
                if numAxes == 3:
                    cc = ss_desc.dimensions_label
                    col_prefix = ss_desc.sensor_type[0] + '_'
                    col_postfix = ' [' + ss_desc.unit + ']'
                    c = [col_prefix + s + col_postfix for s in cc]
                elif numAxes == 1:
                    c = [ss_desc.sensor_type]
                else:
                    print('get_subSensorDataFrame() ERROR: wrong number of sensor axes ({})'.format(numAxes))
            else:
                if numAxes > 0:
                    cc = ss_desc.dimensions_label
                    col_prefix = ss_desc.sensor_type[0] + '_'
                    c = [col_prefix + s for s in cc]
                else:
                    print('get_subSensorDataFrame() ERROR: wrong number of sensor axes ({})'.format(numAxes))
            cols = np.append(cols, c, axis=0)

            cols = np.append(["Time"], cols, axis=0)
            val = np.append(time, data, axis=1)

            ss_data_frame = pd.DataFrame(data=val, columns=cols)

            if labeled:
                tags = self.get_dataStreamTags(sensor_name, sub_sensor_type, sample_start, sample_end)
                if any(bool(tag) for tag in tags):
                    for lbl in self.getAcquisitionLabelClasses():
                        lbl_col = np.zeros(time.shape, dtype=bool)
                        lbl_tags = [x for x in tags if x['Label'] == lbl]
                        for lt in lbl_tags:
                            lbl_col[lt['sample_start']:lt['sample_end']] = True
                        ss_data_frame[lbl] = lbl_col
            return ss_data_frame
        return None

    def get_dataFrame(self, sensor_name, sub_sensor_type, sample_start = 0, sample_end = -1, labeled = False, raw_flag = False):
        ss_desc, ss_stat = self.getSubSensor(sensor_name, ss_type=sub_sensor_type)
        res = self.read_datalog(sensor_name, sub_sensor_type, sample_start, sample_end, raw_flag)
        if res is not None:
            data, time = res
            return self.__to_dataFrame(data, time, ss_desc, sensor_name, sub_sensor_type, labeled, sample_start, sample_end)
        return None
    
    def get_sensorPlot(self, sensor_name, sub_sensor_type, sample_start = 0, sample_end = -1, label=None, subplots=False, raw_flag = False):
        try:
            sub_sensor = self.getSubSensor(sensor_name, None, sub_sensor_type)
            ss_data_frame = None
            if sub_sensor[1].is_active:
                ss_data_frame = self.get_dataFrame(sensor_name, sub_sensor_type, sample_start, sample_end, label is not None, raw_flag)
                ss_desc = sub_sensor[0]

            #Tag columns check (if any)
            if label is not None and len(ss_data_frame.columns) < ss_desc.dimensions + 1:
                click.secho("Warning! No [{}] annotation has been found in the selected acquisition".format(label), fg="yellow")
                label = None

            #Tag columns check (selected label exists?)
            if label is not None and not label in ss_data_frame.columns:
                click.secho("No {} label found in selected acquisition".format(label))
                label = None


            if ss_data_frame is not None:

                    ### labeled and not subplots
                    if label is not None and subplots:
                        cols = ss_desc.dimensions_label
                        fig, axs = plt.subplots(ss_desc.dimensions)
                        if ss_desc.dimensions == 1:
                            axs = (axs,)
                        tit = 'Label: ' + label + '-' + sensor_name + '-' + sub_sensor_type
                        fig.suptitle(tit)
                        
                        for ax in axs:
                            ax.patch.set_facecolor('0.6')
                            ax.patch.set_alpha(float('0.5'))

                        for idx, p in enumerate(axs):
                            true_tag_idxs = ss_data_frame[label].loc[lambda x: x== True].index
                            tag_groups = np.split(true_tag_idxs, np.where(np.diff(true_tag_idxs) != 1)[0]+1)
                            
                            p.plot(ss_data_frame[['Time']], ss_data_frame.iloc[:, idx + 1], color=np.random.rand(3, ))

                            for i in range(len(tag_groups)):
                                start_tag_time = ss_data_frame.at[tag_groups[i][0],'Time']
                                end_tag_time = ss_data_frame.at[tag_groups[i][-1],'Time']
                                p.axvspan(start_tag_time, end_tag_time, facecolor='1', alpha=0.9)
                                p.axvline(x=start_tag_time, color='g', label= str(i) + ") Start " + label)
                                p.axvline(x=end_tag_time, color='r', label= str(i) + ") End " + label)
                            
                            p.set(title=cols[idx])
                        if ss_desc.dimensions > 1:
                            for ax in axs.flat:
                                ax.set(xlabel='Time')
                            for ax in fig.get_axes():
                                ax.label_outer()
                        axs[0].legend(loc='upper left')
                        plt.draw()

                    ### not labeled and not subplots
                    elif label is None and not subplots:
                        cols = ss_desc.dimensions_label
                        plt.figure()
                        for k in range(ss_desc.dimensions):
                            plt.plot(ss_data_frame[['Time']], ss_data_frame.iloc[:, k + 1], color=np.random.rand(3, ), label=cols[k])
                        
                        if not raw_flag:
                            plt.ylabel(ss_desc.unit)

                        plt.title(sensor_name + '-' + ss_desc.sensor_type)
                        plt.xlabel('Time (s)')
                        if ss_desc.sensor_type in ('ACC', 'GYRO', 'MAG', 'MLC'):
                            plt.legend(loc='upper left')
                        plt.draw()

                    ### labeled and not subplots
                    elif label is not None and not subplots:
                        cols = ss_desc.dimensions_label
                        floor = min([min(ss_data_frame.iloc[:, k + 1]) for k in range(ss_desc.dimensions)])
                        fig = plt.figure()
                        ax = fig.add_subplot(111)
                        for k in range(ss_desc.dimensions):
                            not_TAG = np.where(ss_data_frame[label], floor, ss_data_frame.iloc[:, k + 1])
                            TAG = np.where(ss_data_frame[label], ss_data_frame.iloc[:, k + 1], floor)
                            ptag, = ax.plot(ss_data_frame[['Time']], TAG, color=np.random.rand(3, ), label=cols[k])
                            ntag, = ax.plot(ss_data_frame[['Time']], not_TAG, color=np.random.rand(3, ))
                        leg_tag = ax.legend([ptag, ntag], [label, 'not tagged'], loc='upper right')

                        if ss_desc.sensor_type in ('ACC', 'GYRO', 'MAG', 'MLC'):
                            ax.add_artist(leg_tag)
                        if not raw_flag:
                            plt.ylabel(ss_desc.unit)
                        plt.title(sensor_name + '-' + ss_desc.sensor_type)
                        plt.xlabel('Time (s)')
                        plt.draw()

                    ### not labeled and subplots
                    elif label is None and subplots:
                        cols = ss_desc.dimensions_label 
                        fig, axs = plt.subplots(ss_desc.dimensions)
                        if ss_desc.dimensions == 1:
                            axs = (axs,)
                        tit = sensor_name + '-' + ss_desc.sensor_type
                        fig.suptitle(tit)
                        for idx, p in enumerate(axs):
                            p.plot(ss_data_frame[['Time']], ss_data_frame.iloc[:, idx + 1], color=np.random.rand(3, ), label=cols[idx])
                            p.set(title=cols[idx])
                        if ss_desc.dimensions > 1:
                            for ax in axs.flat:
                                ax.set(xlabel = 'Time (s)')
                                if not raw_flag:
                                    ax.set(ylabel = ss_desc.unit)
                            for ax in fig.get_axes():
                                ax.label_outer()
                        else:
                            axs[0].set(xlabel = 'Time (s)')
                            if not raw_flag:
                                axs[0].set(ylabel = ss_desc.unit)
                        plt.draw()
            return None
        except MemoryError:
            click.secho("Memory Error occoured! You should batch process your {} file".format(FM.encode_file_name(sensor_name,sub_sensor_type)), fg='red')
            sys.exit(0)
        except  ValueError:
            click.secho("Value Error occoured! You should batch process your {} file".format(FM.encode_file_name(sensor_name,sub_sensor_type)), fg='red')
            sys.exit(0)

    # #========================================================================================#
    ### OFFLINE CLI Interaction ##############################################################
    #========================================================================================#
    def promptDeviceIdSelect_CLI(self, device_list):
        selected_device = CLI.select_item("Device",device_list)
        selected_device_id = device_list.index(selected_device)
        return selected_device_id

    def promptSensorSelect_CLI(self, sensor_list = None):
        if sensor_list is None:
            sensor_list = self.getSensorList()
        return CLI.select_item("Sensor",sensor_list)

    def promptFileSelect_CLI(self, dat_file_list = None):
        if dat_file_list is None:
            dat_file_list = self.getFileList()
        return CLI.select_item("Data File", dat_file_list)

    def promptLabelSelect_CLI(self, label_list = None):
        if label_list is None:
            label_list = self.getAcquisitionLabelClasses()
        return CLI.select_item("Labels", label_list)
    #========================================================================================#
    ### OFFLINE Unico Data Conversion ########################################################
    #========================================================================================#

    #NOTE Move this in utils.converter package (Based on dataFrame and not on data,time np arrays)
    def to_unico(self, output_folder, sensor_name = 'ISM330DHCX', sample_start = 0, sample_end = -1, use_datalog_tags = False):
        
        if sensor_name != 'ISM330DHCX':
            click.secho('HSDatalog.to_unico: current version supports only ISM330DHCX sensor', fg='red')
            return

        if use_datalog_tags :
            if self.__acq_folder_path is None:
                click.secho("Error. You have to load an acquisition info file", fg='red')
                return
        
        sub_sensors = self.getSubSensors(sensor_name, only_active=True)

        if len(sub_sensors) >= 2 and sub_sensors[1][0].odr != sub_sensors[1][1].odr:
            click.secho('toUnico: {} and {} must have the same odr'.format(sub_sensors[0][0].sensor_type, sub_sensors[0][1].sensor_type), fg= 'red')
            click.secho('{} ODR = {}'.format(sub_sensors[0][0].sensor_type, sub_sensors[1][0].odr), fg='red')
            click.secho('{} ODR = {}'.format(sub_sensors[0][1].sensor_type, sub_sensors[1][1].odr), fg='red')
            exit(0)

        ss_desc_list = sub_sensors[0]

        for ssd in ss_desc_list:
            if ssd.sensor_type == 'ACC':
                axCol = ['A_' + s.upper() + ' [' + ssd.unit + ']' for s in ssd.dimensions_label]
                axData, time = self.read_datalog(sensor_name, ssd.sensor_type, sample_start, sample_end)
                data = axData
                colHeader = axCol
            if ssd.sensor_type == 'GYRO':
                gyCol = ['G_' + s.upper() + ' [' + ssd.unit + ']' for s in ssd.dimensions_label]
                gyData, time = self.read_datalog(sensor_name, ssd.sensor_type, sample_start, sample_end)
                data = gyData
                colHeader = gyCol

        if(len(ss_desc_list) >= 2):
            data = np.concatenate((axData, gyData), axis=1)
            colHeader = axCol + gyCol

        def write_UnicoFile(filename, columns, data: np.array):
            file = open(filename, "w+")
            file.write('STEVAL-STWINKT1 (ISM330DHCX) \n\n\n')
            [file.write(c + '\t') for c in columns]
            file.write('\n')
            for r in range(data.shape[0]):
                for c in range(data.shape[1]):
                    print('{:14.9f}'.format(data[r, c]), end='\t', file=file)
                print('', file=file)
            file.close()

        if use_datalog_tags:

            tags = self.get_dataStreamTags(sensor_name,'ACC', sample_start, sample_end)
            for idx, tag in enumerate(tags):
                labelFileName = tag['Label'] + '_' + sensor_name + '_dataLog_' + str(idx) + '.txt'
                
                tag_sub_folder = os.path.join(output_folder, tag['Label'])
                if not os.path.exists(tag_sub_folder):
                    os.makedirs(tag_sub_folder)
                
                labelFile = os.path.join(tag_sub_folder, labelFileName)
                write_UnicoFile(labelFile, colHeader, data[tag['sample_start']:tag['sample_end'], :])
                click.secho("--> File: \"" + labelFileName + "\" correctly exported" , fg='green')
        else:
            filename = os.path.join(output_folder, sensor_name + '.txt')
            write_UnicoFile(filename, colHeader, data)
            click.secho("--> File: \"" + filename + "\" correctly exported" , fg='green')