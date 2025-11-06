import os
import math
import sys
import xml.etree.ElementTree as ET
import pandas as pd
import pickle
import numpy as np
import xmltodict
import pprint
from rapidfuzz.fuzz import ratio

# initialise config from module on path
current_dir = os.path.dirname(__file__)
module_root = os.path.abspath(os.path.join(current_dir, '../..'))
if module_root not in sys.path:
    sys.path.append(module_root)

from lcpvfxtools import config as cfg

class LensProfileDatabase:
    """
    Class for loading, parsing, and querying Adobe LCP lens profile data.
    """

    def __init__(self, lcp_directory, pickle_file, force_reload=False):
        """
        Initialize the LensProfileDatabase.

        Args:
            lcp_directory (str): Path to the directory containing LCP files.
            pickle_file (str): Path to the pickle file for caching parsed data.
            force_reload (bool): If True, force re-parse LCP files even if pickle exists.
        """
        self.lcp_directory = lcp_directory
        self.pickle_file = pickle_file

        if not force_reload and os.path.exists(self.pickle_file):
            print("Loading cached lens database...")
            self.load_from_pickle()
        else:
            print("Parsing LCP files (this may take a while)...")
            self.data = self._parse_lcp_files()
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', 10)
            pd.set_option('display.width', 1000)
            self.data['Make'] = self.data['Make'].str.strip()
            try:
                self.data['Model'] = self.data['Model'].str.strip()
            except:
                print('Model column not found, skipping strip')
                #self.data['Model'] = ''
                self.data['Model'] = self.data['Make'] # attempting to use make for missing model
            self.data['Lens'] = self.data['Lens'].str.strip()
            self.save_to_pickle()

    def _parse_lcp_files(self):
        """
        Parse all LCP files in the specified directory.

        Returns:
            pd.DataFrame: DataFrame containing all extracted lens profile records.
        """
        records = []
        for root, _, files in os.walk(self.lcp_directory):
            for file in files:
                # if file.endswith("SIGMA (SIGMA 24-105mm F4 DG OS HSM A013) - RAW.lcp"):
                #     path = os.path.join(root, file)
                #     records.extend(self._extract_lcp_data(path))

                if file.endswith(".lcp"):
                    path = os.path.join(root, file)
                    records.extend(self._extract_lcp_data(path))

        return pd.DataFrame(records)

    def _strip_namespace(self, data):
        """
        Recursively remove XML namespaces from dictionary keys.

        Args:
            data (dict or list): XML data as a dictionary or list.

        Returns:
            dict or list: Data with namespaces removed from keys.
        """
        if isinstance(data, dict):
            new_data = {}
            for key, value in data.items():
                new_key = key.split(':')[-1] if ':' in key else key
                new_data[new_key] = self._strip_namespace(value)
            return new_data
        elif isinstance(data, list):
            return [self._strip_namespace(item) for item in data]
        else:
            return data

    def _extract_lcp_data(self, lcp_path):
        """
        Extract relevant lens profile data from a single LCP file.

        Args:
            lcp_path (str): Path to the LCP file.

        Returns:
            list: List of dictionaries, each representing a lens profile record.
        """
        inconsistent_keys = [
            'Model',
            'ImageXCenter',
            'ImageYCenter',
            'FocalLengthX',
            'FocalLengthY'
        ]


        records = []
        with open(lcp_path, 'r') as file:
            xml_data = file.read()
        dict_data = xmltodict.parse(xml_data)
        stripped_dict_data = self._strip_namespace(dict_data)
        li_items = stripped_dict_data['xmpmeta']['RDF']['Description']['CameraProfiles']['Seq']['li']
        for one_item in li_items:

            # Check the type of each value
            if isinstance(one_item, dict):
                record = {}
                record['Model'] = ''
                has_tca = False
                
                for key, value in one_item.items():
                    #print(f'key is {key} and value is {value}')

                        
                    if key == 'Description':
                        
                        if isinstance(value, dict):
                            for desckey, descvalue in value.items():
                                #print(f'desckey is {desckey}')
                                
                                # base description
                                if desckey == 'Make':
                                    #print(f'make is {descvalue}')
                                    record['Make'] = descvalue
                                    if record['Model'] == '':
                                        record['Model'] = descvalue  # if model is empty, use make
                                elif desckey == 'Model':
                                    #print(f'model is {descvalue}')
                                    record['Model'] = descvalue
                                elif desckey == 'Lens':
                                    #print(f'lens is {descvalue}')
                                    record['Lens'] = descvalue
                                elif desckey == 'FocalLength':
                                    record['FocalLength'] = float(descvalue)
                                elif desckey == 'FocusDistance':
                                    record['FocusDistance'] = float(descvalue)
                                elif desckey == 'ApertureValue':
                                    record['ApertureValue'] = float(descvalue)
                                elif desckey == 'LensPrettyName':
                                    record['LensPrettyName'] = descvalue
                        
                    
                                # perspective model
                                elif desckey == 'PerspectiveModel':
                                    #print(f'perspective descvalue is {descvalue}')
                                    if isinstance(descvalue, dict):
                                        for desckey2, descvalue2 in descvalue.items():
                                            if desckey2 == 'PerspectiveModel':
                                                for desckey3, descvalue3 in descvalue2.items():
                                                    if desckey3 == 'Description':
                                                        if isinstance(descvalue3, dict):
                                                            for pmkey, pmvalue in descvalue3.items():
                                                    
                                                                if pmkey == 'ImageXCenter':
                                                                    #print(pmvalue)
                                                                    record['ImageXCenter'] = float(pmvalue)
                                                                elif pmkey == 'ImageYCenter':
                                                                    #print(pmvalue)
                                                                    record['ImageYCenter'] = float(pmvalue)
                                                                elif pmkey == 'FocalLengthX':
                                                                    #print(pmvalue)
                                                                    record['FocalLengthX'] = float(pmvalue)
                                                                elif pmkey == 'FocalLengthY':
                                                                    #print(pmvalue)
                                                                    record['FocalLengthY'] = float(pmvalue)
                                                                elif pmkey == 'RadialDistortParam1':
                                                                    #print(pmvalue)
                                                                    if 'RadialDistortParam1' not in record:
                                                                        #print('radial 1')
                                                                        record['RadialDistortParam1'] = float(pmvalue)
                                                                elif pmkey == 'RadialDistortParam2':
                                                                    #print(pmvalue)
                                                                    if 'RadialDistortParam2' not in record:
                                                                        #print('radial 2')
                                                                        record['RadialDistortParam2'] = float(pmvalue)
                                                                elif pmkey == 'RadialDistortParam3':
                                                                    #print(pmvalue)
                                                                    if 'RadialDistortParam3' not in record:
                                                                        #print('radial 3')
                                                                        record['RadialDistortParam3'] = float(pmvalue)
                                                                elif pmkey == 'VignetteModel':
                                                                    #print(f'pmkey is {pmkey} and pmvalue is {pmvalue}')
                                                                    if isinstance(pmvalue, dict):
                                                                        for vmkey, vmvalue in pmvalue.items():
                                                                            #print(f'vmkey is {vmkey} and vmvalue is {vmvalue}')
                                                                            if vmkey == 'VignetteModelParam1':
                                                                                record['VignetteModelParam1'] = float(vmvalue)
                                                                            elif vmkey == 'VignetteModelParam2':
                                                                                record['VignetteModelParam2'] = float(vmvalue)
                                                                            elif vmkey == 'VignetteModelParam3':
                                                                                record['VignetteModelParam3'] = float(vmvalue)
                                                                    else:
                                                                        print('vignette not a dict')
                        
                    
                                                                elif pmkey == 'ChromaticRedGreenModel':
                                                                    #print('chroma 1')
                                                                    if isinstance(pmvalue, dict):
                                                                        for crgkey, crgvalue in pmvalue.items():
                                                                            if crgkey == 'RadialDistortParam1':
                                                                                record['TCA_RedGreen_Radial1'] = float(crgvalue)
                                                                            elif crgkey == 'RadialDistortParam2':
                                                                                record['TCA_RedGreen_Radial2'] = float(crgvalue)
                                                                            elif crgkey == 'RadialDistortParam3':
                                                                                record['TCA_RedGreen_Radial3'] = float(crgvalue)
                                                                            elif crgkey == 'ScaleFactor':
                                                                                record['TCA_RedGreen_ScaleFactor'] = float(crgvalue)
                    
                                                                elif pmkey == 'ChromaticGreenModel':
                                                                    if isinstance(pmvalue, dict):
                                                                        for cgkey, cgvalue in pmvalue.items():
                                                                            if cgkey == 'RadialDistortParam1':
                                                                                record['TCA_Green_Radial1'] = float(cgvalue)
                                                                            elif cgkey == 'RadialDistortParam2':
                                                                                record['TCA_Green_Radial2'] = float(cgvalue)
                                                                            elif cgkey == 'RadialDistortParam3':
                                                                                record['TCA_Green_Radial3'] = float(cgvalue)
                                                                            elif cgkey == 'ScaleFactor':
                                                                                record['TCA_Green_ScaleFactor'] = float(cgvalue)
                                                                                
                                                                elif pmkey == 'ChromaticBlueGreenModel':
                                                                    if isinstance(pmvalue, dict):
                                                                        for cbgkey, cbgvalue in pmvalue.items():
                                                                            if cbgkey == 'RadialDistortParam1':
                                                                                record['TCA_BlueGreen_Radial1'] = float(cbgvalue)
                                                                            elif cbgkey == 'RadialDistortParam2':
                                                                                record['TCA_BlueGreen_Radial2'] = float(cbgvalue)
                                                                            elif cbgkey == 'RadialDistortParam3':
                                                                                record['TCA_BlueGreen_Radial3'] = float(cbgvalue)
                                                                            elif cbgkey == 'ScaleFactor':
                                                                                record['TCA_BlueGreen_ScaleFactor'] = float(cbgvalue)
                                                        
                                                                else:
                                                                    #print(f'pmkey exception of {pmkey} and {pmvalue}')
                                                                    pass
                                                    
                                                    else:
                                                        if isinstance(descvalue3, str):
                                                            pmkey = desckey3
                                                            pmvalue = descvalue3

                                                            if pmkey == 'ImageXCenter':
                                                                #print(pmvalue)
                                                                record['ImageXCenter'] = float(pmvalue)
                                                            elif pmkey == 'ImageYCenter':
                                                                #print(pmvalue)
                                                                record['ImageYCenter'] = float(pmvalue)
                                                            elif pmkey == 'FocalLengthX':
                                                                #print(pmvalue)
                                                                record['FocalLengthX'] = float(pmvalue)
                                                            elif pmkey == 'FocalLengthY':
                                                                #print(pmvalue)
                                                                record['FocalLengthY'] = float(pmvalue)
                                                            elif pmkey == 'RadialDistortParam1':
                                                                #print(pmvalue)
                                                                if 'RadialDistortParam1' not in record:
                                                                    #print('radial 1')
                                                                    record['RadialDistortParam1'] = float(pmvalue)
                                                            elif pmkey == 'RadialDistortParam2':
                                                                #print(pmvalue)
                                                                if 'RadialDistortParam2' not in record:
                                                                    #print('radial 2')
                                                                    record['RadialDistortParam2'] = float(pmvalue)
                                                            elif pmkey == 'RadialDistortParam3':
                                                                #print(pmvalue)
                                                                if 'RadialDistortParam3' not in record:
                                                                    #print('radial 3')
                                                                    record['RadialDistortParam3'] = float(pmvalue)
                                                            else:
                                                                print(f'pmkey is {pmkey} and pmvalue is {pmvalue}')
                                                                #pass
                                                        else:
                                                            #print('string exception')
                                                            pass
                                            elif desckey2 == 'Description':
                                                if isinstance(descvalue2, dict):
                                                    for pmkey, pmvalue in descvalue2.items():
                                            
                                                        if pmkey == 'ImageXCenter':
                                                            #print(pmvalue)
                                                            record['ImageXCenter'] = float(pmvalue)
                                                        elif pmkey == 'ImageYCenter':
                                                            #print(pmvalue)
                                                            record['ImageYCenter'] = float(pmvalue)
                                                        elif pmkey == 'FocalLengthX':
                                                            #print(pmvalue)
                                                            record['FocalLengthX'] = float(pmvalue)
                                                        elif pmkey == 'FocalLengthY':
                                                            #print(pmvalue)
                                                            record['FocalLengthY'] = float(pmvalue)
                                                        elif pmkey == 'RadialDistortParam1':
                                                            #print(pmvalue)
                                                            if 'RadialDistortParam1' not in record:
                                                                #print('radial 1')
                                                                record['RadialDistortParam1'] = float(pmvalue)
                                                        elif pmkey == 'RadialDistortParam2':
                                                            #print(pmvalue)
                                                            if 'RadialDistortParam2' not in record:
                                                                #print('radial 2')
                                                                record['RadialDistortParam2'] = float(pmvalue)
                                                        elif pmkey == 'RadialDistortParam3':
                                                            #print(pmvalue)
                                                            if 'RadialDistortParam3' not in record:
                                                                #print('radial 3')
                                                                record['RadialDistortParam3'] = float(pmvalue)
                                                        elif pmkey == 'VignetteModel':
                                                            #print(f'pmkey is {pmkey} and pmvalue is {pmvalue}')
                                                            if isinstance(pmvalue, dict):
                                                                for vmkey, vmvalue in pmvalue.items():
                                                                    #print(f'vmkey is {vmkey} and vmvalue is {vmvalue}')
                                                                    if vmkey == 'VignetteModelParam1':
                                                                        record['VignetteModelParam1'] = float(vmvalue)
                                                                    elif vmkey == 'VignetteModelParam2':
                                                                        record['VignetteModelParam2'] = float(vmvalue)
                                                                    elif vmkey == 'VignetteModelParam3':
                                                                        record['VignetteModelParam3'] = float(vmvalue)
                                                                    elif vmkey == 'Description':
                                                                        if isinstance(vmvalue, dict):
                                                                            
                                                                            for vmkey2, vmvalue2 in vmvalue.items():
                                                                                if vmkey2 == 'VignetteModelParam1':
                                                                                    record['VignetteModelParam1'] = float(vmvalue2)
                                                                                elif vmkey2 == 'VignetteModelParam2':
                                                                                    record['VignetteModelParam2'] = float(vmvalue2)
                                                                                elif vmkey2 == 'VignetteModelParam3':
                                                                                    record['VignetteModelParam3'] = float(vmvalue2)
                                                                    else:
                                                                        #print(f'vignette exception of {vmkey}')
                                                                        pass
                                                            else:
                                                                #print('vignette not a dict')
                                                                pass
                
            
                                                        elif pmkey == 'ChromaticRedGreenModel':
                                                            #print('chroma 2')
                                                            #has_tca = True
                                                            if isinstance(pmvalue, dict):
                                                                for crgkey, crgvalue in pmvalue.items():
                                                                    if crgkey == 'RadialDistortParam1':
                                                                        #print('made it into tca')
                                                                        record['TCA_RedGreen_Radial1'] = float(crgvalue)
                                                                    elif crgkey == 'RadialDistortParam2':
                                                                        record['TCA_RedGreen_Radial2'] = float(crgvalue)
                                                                    elif crgkey == 'RadialDistortParam3':
                                                                        record['TCA_RedGreen_Radial3'] = float(crgvalue)
                                                                    elif crgkey == 'ScaleFactor':
                                                                        record['TCA_RedGreen_ScaleFactor'] = float(crgvalue)
                                                                    else:
                                                                        #print(f'exception chroma key was {crgkey} and {crgvalue}')
                                                                        pass
                                                            else:
                                                                #print('chroma 2 not a dict')
                                                                pass
                                                        elif pmkey == 'ChromaticGreenModel':
                                                            if isinstance(pmvalue, dict):
                                                                for cgkey, cgvalue in pmvalue.items():
                                                                    if cgkey == 'RadialDistortParam1':
                                                                        record['TCA_Green_Radial1'] = float(cgvalue)
                                                                    elif cgkey == 'RadialDistortParam2':
                                                                        record['TCA_Green_Radial2'] = float(cgvalue)
                                                                    elif cgkey == 'RadialDistortParam3':
                                                                        record['TCA_Green_Radial3'] = float(cgvalue)
                                                                    elif cgkey == 'ScaleFactor':
                                                                        record['TCA_Green_ScaleFactor'] = float(cgvalue)
                                                                        
                                                        elif pmkey == 'ChromaticBlueGreenModel':
                                                            if isinstance(pmvalue, dict):
                                                                for cbgkey, cbgvalue in pmvalue.items():
                                                                    if cbgkey == 'RadialDistortParam1':
                                                                        record['TCA_BlueGreen_Radial1'] = float(cbgvalue)
                                                                    elif cbgkey == 'RadialDistortParam2':
                                                                        record['TCA_BlueGreen_Radial2'] = float(cbgvalue)
                                                                    elif cbgkey == 'RadialDistortParam3':
                                                                        record['TCA_BlueGreen_Radial3'] = float(cbgvalue)
                                                                    elif cbgkey == 'ScaleFactor':
                                                                        record['TCA_BlueGreen_ScaleFactor'] = float(cbgvalue)
                                                
                                                        else:

                                                            #print(f'pmkey exception of {pmkey} and {pmvalue}')
                                                            pass
                                            
                                                else:
                                                    #print('description not a dict')
                                                    pass
                                            
                                            else:
                                                #print(f'desckey2 is {desckey2} and descvalue2 is {descvalue2}')
                                                pmkey = desckey2
                                                pmvalue = descvalue2
                                                if pmkey == 'ImageXCenter':
                                                    #print(pmvalue)
                                                    record['ImageXCenter'] = float(pmvalue)
                                                elif pmkey == 'ImageYCenter':
                                                    #print(pmvalue)
                                                    record['ImageYCenter'] = float(pmvalue)
                                                elif pmkey == 'FocalLengthX':
                                                    #print(pmvalue)
                                                    record['FocalLengthX'] = float(pmvalue)
                                                elif pmkey == 'FocalLengthY':
                                                    #print(pmvalue)
                                                    record['FocalLengthY'] = float(pmvalue)
                                                elif pmkey == 'RadialDistortParam1':
                                                    #print(pmvalue)
                                                    if 'RadialDistortParam1' not in record:
                                                        record['RadialDistortParam1'] = float(pmvalue)
                                                elif pmkey == 'RadialDistortParam2':
                                                    #print(pmvalue)
                                                    if 'RadialDistortParam2' not in record:
                                                        record['RadialDistortParam2'] = float(pmvalue)
                                                elif pmkey == 'RadialDistortParam3':
                                                    #print(pmvalue)
                                                    if 'RadialDistortParam3' not in record:
                                                        record['RadialDistortParam3'] = float(pmvalue)
                                                elif pmkey == 'VignetteModel':
                                                    #print(f'pmkey is {pmkey} and pmvalue is {pmvalue}')
                                                    if isinstance(pmvalue, dict):
                                                        for vmkey, vmvalue in pmvalue.items():
                                                            #print(f'vmkey is {vmkey} and vmvalue is {vmvalue}')
                                                            if vmkey == 'VignetteModelParam1':
                                                                record['VignetteModelParam1'] = float(vmvalue)
                                                            elif vmkey == 'VignetteModelParam2':
                                                                record['VignetteModelParam2'] = float(vmvalue)
                                                            elif vmkey == 'VignetteModelParam3':
                                                                record['VignetteModelParam3'] = float(vmvalue)
                                                    else:
                                                        #print('vignette not a dict')
                                                        pass
    
                                                elif pmkey == 'ChromaticRedGreenModel':
                                                    #print('chroma 3')
                                                    if isinstance(pmvalue, dict):
                                                        for crgkey, crgvalue in pmvalue.items():
                                                            if crgkey == 'RadialDistortParam1':
                                                                record['TCA_RedGreen_Radial1'] = float(crgvalue)
                                                            elif crgkey == 'RadialDistortParam2':
                                                                record['TCA_RedGreen_Radial2'] = float(crgvalue)
                                                            elif crgkey == 'RadialDistortParam3':
                                                                record['TCA_RedGreen_Radial3'] = float(crgvalue)
                                                            elif crgkey == 'ScaleFactor':
                                                                record['TCA_RedGreen_ScaleFactor'] = float(crgvalue)
    
                                                elif pmkey == 'ChromaticGreenModel':
                                                    if isinstance(pmvalue, dict):
                                                        for cgkey, cgvalue in pmvalue.items():
                                                            if cgkey == 'RadialDistortParam1':
                                                                record['TCA_Green_Radial1'] = float(cgvalue)
                                                            elif cgkey == 'RadialDistortParam2':
                                                                record['TCA_Green_Radial2'] = float(cgvalue)
                                                            elif cgkey == 'RadialDistortParam3':
                                                                record['TCA_Green_Radial3'] = float(cgvalue)
                                                            elif cgkey == 'ScaleFactor':
                                                                record['TCA_Green_ScaleFactor'] = float(cgvalue)
                                                                
                                                elif pmkey == 'ChromaticBlueGreenModel':
                                                    if isinstance(pmvalue, dict):
                                                        for cbgkey, cbgvalue in pmvalue.items():
                                                            if cbgkey == 'RadialDistortParam1':
                                                                record['TCA_BlueGreen_Radial1'] = float(cbgvalue)
                                                            elif cbgkey == 'RadialDistortParam2':
                                                                record['TCA_BlueGreen_Radial2'] = float(cbgvalue)
                                                            elif cbgkey == 'RadialDistortParam3':
                                                                record['TCA_BlueGreen_Radial3'] = float(cbgvalue)
                                                            elif cbgkey == 'ScaleFactor':
                                                                record['TCA_BlueGreen_ScaleFactor'] = float(cbgvalue)
                                                else:
                                                    #print(f'pmkey exception with {pmkey} and {pmvalue}')
                                                    pass
                                    else:
                                        #print('perspective model value not a dict')
                                        pass
                                else:
                                    '''
                                    if 'odel' in desckey:
                                        print(f'desckey exception is {desckey}')
                                    '''
                                    pass
                        else:
                            #print(f'value dict exception is {value}')
                            pass
                    else:
                        '''
                        print(f'key exception is {key}')
                        print(f'value exception is {value}')
                        pass
                        '''

                        desckey = key
                        descvalue = value

                        if desckey == 'Make':
                            #print(f'make is {descvalue}')
                            record['Make'] = descvalue
                            if record['Model'] == '':
                                record['Model'] = descvalue # attempting to use make for missing model
                        elif desckey == 'Model':
                            #print(f'model is {descvalue}')
                            record['Model'] = descvalue
                        elif desckey == 'Lens':
                            #print(f'lens is {descvalue}')
                            record['Lens'] = descvalue
                        elif desckey == 'FocalLength':
                            record['FocalLength'] = float(descvalue)
                        elif desckey == 'FocusDistance':
                            record['FocusDistance'] = float(descvalue)
                        elif desckey == 'ApertureValue':
                            record['ApertureValue'] = float(descvalue)
                        elif desckey == 'LensPrettyName':
                            record['LensPrettyName'] = descvalue
                
            
                        # perspective model
                        elif desckey == 'PerspectiveModel':
                            #print(f'perspective descvalue is {descvalue}')
                            if isinstance(descvalue, dict):
                                for desckey2, descvalue2 in descvalue.items():
                                    if desckey2 == 'PerspectiveModel':
                                        for desckey3, descvalue3 in descvalue2.items():
                                            if desckey3 == 'Description':
                                                if isinstance(descvalue3, dict):
                                                    for pmkey, pmvalue in descvalue3.items():
                                            
                                                        if pmkey == 'ImageXCenter':
                                                            #print(pmvalue)
                                                            record['ImageXCenter'] = float(pmvalue)
                                                        elif pmkey == 'ImageYCenter':
                                                            #print(pmvalue)
                                                            record['ImageYCenter'] = float(pmvalue)
                                                        elif pmkey == 'FocalLengthX':
                                                            #print(pmvalue)
                                                            record['FocalLengthX'] = float(pmvalue)
                                                        elif pmkey == 'FocalLengthY':
                                                            #print(pmvalue)
                                                            record['FocalLengthY'] = float(pmvalue)
                                                        elif pmkey == 'RadialDistortParam1':
                                                            #print(pmvalue)
                                                            if 'RadialDistortParam1' not in record:
                                                                record['RadialDistortParam1'] = float(pmvalue)
                                                        elif pmkey == 'RadialDistortParam2':
                                                            #print(pmvalue)
                                                            if 'RadialDistortParam2' not in record:
                                                                record['RadialDistortParam2'] = float(pmvalue)
                                                        elif pmkey == 'RadialDistortParam3':
                                                            #print(pmvalue)
                                                            if 'RadialDistortParam3' not in record:
                                                                record['RadialDistortParam3'] = float(pmvalue)
                                                        elif pmkey == 'VignetteModel':
                                                            #print(f'pmkey is {pmkey} and pmvalue is {pmvalue}')
                                                            if isinstance(pmvalue, dict):
                                                                for vmkey, vmvalue in pmvalue.items():
                                                                    #print(f'vmkey is {vmkey} and vmvalue is {vmvalue}')
                                                                    if vmkey == 'VignetteModelParam1':
                                                                        record['VignetteModelParam1'] = float(vmvalue)
                                                                    elif vmkey == 'VignetteModelParam2':
                                                                        record['VignetteModelParam2'] = float(vmvalue)
                                                                    elif vmkey == 'VignetteModelParam3':
                                                                        record['VignetteModelParam3'] = float(vmvalue)
                                                            else:
                                                                print('vignette not a dict')
                
            
                                                        elif pmkey == 'ChromaticRedGreenModel':
                                                            #print('chroma 1')
                                                            if isinstance(pmvalue, dict):
                                                                for crgkey, crgvalue in pmvalue.items():
                                                                    if crgkey == 'RadialDistortParam1':
                                                                        record['TCA_RedGreen_Radial1'] = float(crgvalue)
                                                                    elif crgkey == 'RadialDistortParam2':
                                                                        record['TCA_RedGreen_Radial2'] = float(crgvalue)
                                                                    elif crgkey == 'RadialDistortParam3':
                                                                        record['TCA_RedGreen_Radial3'] = float(crgvalue)
                                                                    elif crgkey == 'ScaleFactor':
                                                                        record['TCA_RedGreen_ScaleFactor'] = float(crgvalue)
            
                                                        elif pmkey == 'ChromaticGreenModel':
                                                            if isinstance(pmvalue, dict):
                                                                for cgkey, cgvalue in pmvalue.items():
                                                                    if cgkey == 'RadialDistortParam1':
                                                                        record['TCA_Green_Radial1'] = float(cgvalue)
                                                                    elif cgkey == 'RadialDistortParam2':
                                                                        record['TCA_Green_Radial2'] = float(cgvalue)
                                                                    elif cgkey == 'RadialDistortParam3':
                                                                        record['TCA_Green_Radial3'] = float(cgvalue)
                                                                    elif cgkey == 'ScaleFactor':
                                                                        record['TCA_Green_ScaleFactor'] = float(cgvalue)
                                                                        
                                                        elif pmkey == 'ChromaticBlueGreenModel':
                                                            if isinstance(pmvalue, dict):
                                                                for cbgkey, cbgvalue in pmvalue.items():
                                                                    if cbgkey == 'RadialDistortParam1':
                                                                        record['TCA_BlueGreen_Radial1'] = float(cbgvalue)
                                                                    elif cbgkey == 'RadialDistortParam2':
                                                                        record['TCA_BlueGreen_Radial2'] = float(cbgvalue)
                                                                    elif cbgkey == 'RadialDistortParam3':
                                                                        record['TCA_BlueGreen_Radial3'] = float(cbgvalue)
                                                                    elif cbgkey == 'ScaleFactor':
                                                                        record['TCA_BlueGreen_ScaleFactor'] = float(cbgvalue)
                                                
                                                        else:
                                                            #print(f'pmkey exception of {pmkey} and {pmvalue}')
                                                            pass
                                            
                                            else:
                                                if isinstance(descvalue3, str):
                                                    pmkey = desckey3
                                                    pmvalue = descvalue3

                                                    if pmkey == 'ImageXCenter':
                                                        #print(pmvalue)
                                                        record['ImageXCenter'] = float(pmvalue)
                                                    elif pmkey == 'ImageYCenter':
                                                        #print(pmvalue)
                                                        record['ImageYCenter'] = float(pmvalue)
                                                    elif pmkey == 'FocalLengthX':
                                                        #print(pmvalue)
                                                        record['FocalLengthX'] = float(pmvalue)
                                                    elif pmkey == 'FocalLengthY':
                                                        #print(pmvalue)
                                                        record['FocalLengthY'] = float(pmvalue)
                                                    elif pmkey == 'RadialDistortParam1':
                                                        #print(pmvalue)
                                                        if 'RadialDistortParam1' not in record:
                                                            #print('radial 1')
                                                            record['RadialDistortParam1'] = float(pmvalue)
                                                    elif pmkey == 'RadialDistortParam2':
                                                        #print(pmvalue)
                                                        if 'RadialDistortParam2' not in record:
                                                            #print('radial 2')
                                                            record['RadialDistortParam2'] = float(pmvalue)
                                                    elif pmkey == 'RadialDistortParam3':
                                                        #print(pmvalue)
                                                        if 'RadialDistortParam3' not in record:
                                                            #print('radial 3')
                                                            record['RadialDistortParam3'] = float(pmvalue)
                                                    else:
                                                        print(f'pmkey is {pmkey} and pmvalue is {pmvalue}')
                                                        #pass
                                                else:
                                                    #print('string exception')
                                                    pass
                                    else:
                                        # this is likely the nikon section
                                        if isinstance(descvalue2, dict):
                                            pmkey = desckey2
                                            pmvalue = descvalue2

                                            if pmkey == 'VignetteModel':
                                                #print(f'pmkey is {pmkey} and pmvalue is {pmvalue} and pmvalue is {type(pmvalue)}')
                                                if isinstance(pmvalue, dict):
                                                    for vmkey, vmvalue in pmvalue.items():
                                                        #print('made it here')
                                                        #print(f'vmkey is {vmkey} and vmvalue is {vmvalue}')
                                                        if vmkey == 'VignetteModelParam1':
                                                            record['VignetteModelParam1'] = float(vmvalue)
                                                        elif vmkey == 'VignetteModelParam2':
                                                            record['VignetteModelParam2'] = float(vmvalue)
                                                        elif vmkey == 'VignetteModelParam3':
                                                            record['VignetteModelParam3'] = float(vmvalue)
                                                else:
                                                    print('vignette not a dict')
    

                                            elif pmkey == 'ChromaticRedGreenModel':
                                                #print('chroma 1')
                                                if isinstance(pmvalue, dict):
                                                    for crgkey, crgvalue in pmvalue.items():
                                                        if crgkey == 'RadialDistortParam1':
                                                            record['TCA_RedGreen_Radial1'] = float(crgvalue)
                                                        elif crgkey == 'RadialDistortParam2':
                                                            record['TCA_RedGreen_Radial2'] = float(crgvalue)
                                                        elif crgkey == 'RadialDistortParam3':
                                                            record['TCA_RedGreen_Radial3'] = float(crgvalue)
                                                        elif crgkey == 'ScaleFactor':
                                                            record['TCA_RedGreen_ScaleFactor'] = float(crgvalue)

                                            elif pmkey == 'ChromaticGreenModel':
                                                if isinstance(pmvalue, dict):
                                                    for cgkey, cgvalue in pmvalue.items():
                                                        if cgkey == 'RadialDistortParam1':
                                                            record['TCA_Green_Radial1'] = float(cgvalue)
                                                        elif cgkey == 'RadialDistortParam2':
                                                            record['TCA_Green_Radial2'] = float(cgvalue)
                                                        elif cgkey == 'RadialDistortParam3':
                                                            record['TCA_Green_Radial3'] = float(cgvalue)
                                                        elif cgkey == 'ScaleFactor':
                                                            record['TCA_Green_ScaleFactor'] = float(cgvalue)
                                                            
                                            elif pmkey == 'ChromaticBlueGreenModel':
                                                if isinstance(pmvalue, dict):
                                                    for cbgkey, cbgvalue in pmvalue.items():
                                                        if cbgkey == 'RadialDistortParam1':
                                                            record['TCA_BlueGreen_Radial1'] = float(cbgvalue)
                                                        elif cbgkey == 'RadialDistortParam2':
                                                            record['TCA_BlueGreen_Radial2'] = float(cbgvalue)
                                                        elif cbgkey == 'RadialDistortParam3':
                                                            record['TCA_BlueGreen_Radial3'] = float(cbgvalue)
                                                        elif cbgkey == 'ScaleFactor':
                                                            record['TCA_BlueGreen_ScaleFactor'] = float(cbgvalue)
                                    
                                            else:
                                                #print(f'pmkey exception of {pmkey} and {pmvalue}')
                                                pass

                                        else:
                                            pmkey = desckey2
                                            pmvalue = descvalue2
                                            if pmkey == 'ImageXCenter':
                                                #print(pmvalue)
                                                record['ImageXCenter'] = float(pmvalue)
                                            elif pmkey == 'ImageYCenter':
                                                #print(pmvalue)
                                                record['ImageYCenter'] = float(pmvalue)
                                            elif pmkey == 'FocalLengthX':
                                                #print(pmvalue)
                                                record['FocalLengthX'] = float(pmvalue)
                                            elif pmkey == 'FocalLengthY':
                                                #print(pmvalue)
                                                record['FocalLengthY'] = float(pmvalue)
                                            elif pmkey == 'RadialDistortParam1':
                                                #print(pmvalue)
                                                if 'RadialDistortParam1' not in record:
                                                    #print('radial 1')
                                                    record['RadialDistortParam1'] = float(pmvalue)
                                            elif pmkey == 'RadialDistortParam2':
                                                #print(pmvalue)
                                                if 'RadialDistortParam2' not in record:
                                                    #print('radial 2')
                                                    record['RadialDistortParam2'] = float(pmvalue)
                                            elif pmkey == 'RadialDistortParam3':
                                                #print(pmvalue)
                                                if 'RadialDistortParam3' not in record:
                                                    #print('radial 3')
                                                    record['RadialDistortParam3'] = float(pmvalue)
                            else:
                                print(f'exception: desckey is {desckey} and descvalue is {descvalue}')

                                pass

                        else:
                            #print(f'key exception is {desckey} and value is {descvalue}')
                            pass

                if record != {}:
                    # tidying
                    if record['Make'] == '':
                        record['Make'] = record['Model']  # if make is empty, use model

                    for one_key in inconsistent_keys:
                        if one_key not in record:
                            record[one_key] = None



                    records.append(record)
                    #print(record)
                
            
        
            else:
                #print(f'non-dict is {one_item} and type is {type(one_item)}')
                pass
            
            
        return records

    def save_to_pickle(self):
        """
        Save the parsed lens profile data to a pickle file.
        """
        with open(self.pickle_file, 'wb') as f:
            pickle.dump(self.data, f)

    def load_from_pickle(self):
        """
        Load lens profile data from a pickle file.
        """
        with open(self.pickle_file, 'rb') as f:
            self.data = pickle.load(f)

    def find_lens_data(self, make, model, lens):
        """
        Find lens data for the given make, model, and lens.

        Args:
            make (str): Camera make.
            model (str): Camera model.
            lens (str): Lens model.

        Returns:
            pd.DataFrame: DataFrame filtered for the specified make, model, and lens.
        """
        makedf = self.data[self.data['Make'] == make]
        modeldf = makedf[makedf['Model'] == model]
        return modeldf[modeldf['Lens'] == lens]

    def get_exact_match(self, df, focal_length, focus_distance, aperture_value):
        """
        Find an exact match for focal length, focus distance, and aperture value.

        Args:
            df (pd.DataFrame): DataFrame of candidate profiles.
            focal_length (float): Focal length to match.
            focus_distance (float): Focus distance to match.
            aperture_value (float): Aperture value to match.

        Returns:
            pd.DataFrame: DataFrame of exact matches.
        """
        return df[(df["FocalLength"] == focal_length) & (df["FocusDistance"] == focus_distance) & (df["ApertureValue"] == aperture_value)]

    def interpolate_values(self, df, focal_length, focus_distance, aperture_value):
        """
        Interpolate values for TCA, vignetting, and distortion parameters.

        Args:
            df (pd.DataFrame): DataFrame of candidate profiles.
            focal_length (float): Focal length for interpolation.
            focus_distance (float): Focus distance for interpolation.
            aperture_value (float): Aperture value for interpolation.

        Returns:
            dict: Dictionary of interpolated parameter values.
        """
        print(f"Interpolating values for FocalLength: {focal_length}, FocusDistance: {focus_distance}, ApertureValue: {aperture_value}")

        df_sorted = df.sort_values(["FocalLength", "FocusDistance", "ApertureValue"])

        # Check if the input values are within the bounds of the data
        if focal_length < df_sorted["FocalLength"].min() or focal_length > df_sorted["FocalLength"].max() or \
           focus_distance < df_sorted["FocusDistance"].min() or focus_distance > df_sorted["FocusDistance"].max() or \
           aperture_value < df_sorted["ApertureValue"].min() or aperture_value > df_sorted["ApertureValue"].max():
            print("Input values are out of bounds.")
            return None  # Input values are out of bounds

        def trilinear_interpolate(x1, x2, y1, y2, z1, z2, c000, c001, c010, c011, c100, c101, c110, c111, x, y, z):
            """Perform trilinear interpolation with handling for identical bounds."""
            # Handle identical bounds for x (focal length)
            if x1 == x2:
                c00 = c000
                c01 = c001
                c10 = c010
                c11 = c011
            else:
                c00 = ((x2 - x) * c000 + (x - x1) * c100) / (x2 - x1)
                c01 = ((x2 - x) * c001 + (x - x1) * c101) / (x2 - x1)
                c10 = ((x2 - x) * c010 + (x - x1) * c110) / (x2 - x1)
                c11 = ((x2 - x) * c011 + (x - x1) * c111) / (x2 - x1)

            # Handle identical bounds for y (focus distance)
            if y1 == y2:
                c0 = c00
                c1 = c01
            else:
                c0 = ((y2 - y) * c00 + (y - y1) * c10) / (y2 - y1)
                c1 = ((y2 - y) * c01 + (y - y1) * c11) / (y2 - y1)

            # Handle identical bounds for z (aperture value)
            if z1 == z2:
                return c0
            else:
                return ((z2 - z) * c0 + (z - z1) * c1) / (z2 - z1)

        params = [
            "RadialDistortParam1", "RadialDistortParam2", "RadialDistortParam3",
            "VignetteModelParam1", "VignetteModelParam2", "VignetteModelParam3",
            "TCA_RedGreen_Radial1", "TCA_RedGreen_Radial2", "TCA_RedGreen_Radial3",
            "TCA_Green_Radial1", "TCA_Green_Radial2", "TCA_Green_Radial3",
            "TCA_BlueGreen_Radial1", "TCA_BlueGreen_Radial2", "TCA_BlueGreen_Radial3"
        ]

        interpolated = {}
        for param in params:
            # Skip interpolation if the parameter does not exist in the DataFrame
            if param not in df.columns:
                interpolated[param] = None
                print(f"Parameter {param} not found in DataFrame.")
                continue

            # Filter bounds for focal length
            focal_lower_candidates = df_sorted[df_sorted["FocalLength"] <= focal_length]
            focal_lower_candidates = focal_lower_candidates[focal_lower_candidates[param].notna()]
            for one_candidate in focal_lower_candidates['FocalLength']: 
                #print(one_candidate)
                pass
            focal_lower = focal_lower_candidates.tail(1)
            #print(f'focal lower bound of {focal_lower["FocalLength"].values[0]} is at or below {focal_length}')

            focal_upper_candidates = df_sorted[df_sorted["FocalLength"] >= focal_length]
            focal_upper_candidates = focal_upper_candidates[focal_upper_candidates[param].notna()]
            for one_candidate in focal_upper_candidates['FocalLength']: 
                #print(one_candidate)
                pass
            focal_upper = focal_upper_candidates.head(1)
            #print(f'focal upper bound of {focal_upper["FocalLength"].values[0]} is above {focal_length}')

            # Filter bounds for focus distance
            focus_lower_candidates = df_sorted[df_sorted["FocusDistance"] <= focus_distance]
            focus_lower_candidates = focus_lower_candidates[focus_lower_candidates[param].notna()]
            focus_lower = focus_lower_candidates.tail(1)
            #print(f'focus lower bound of {focus_lower["FocusDistance"].values[0]} is at or below {focus_distance}')
            focus_upper_candidates = df_sorted[df_sorted["FocusDistance"] >= focus_distance]
            focus_upper_candidates = focus_upper_candidates[focus_upper_candidates[param].notna()]
            focus_upper = focus_upper_candidates.head(1)
            #print(f'focus upper bound of {focus_upper["FocusDistance"].values[0]} is above {focus_distance}')
            # Filter bounds for aperture value
            aperture_lower_candidates = df_sorted[df_sorted["ApertureValue"] <= aperture_value]
            aperture_lower_candidates = aperture_lower_candidates[aperture_lower_candidates[param].notna()]
            aperture_lower = aperture_lower_candidates.tail(1)
            #print(f'aperture lower bound of {aperture_lower["ApertureValue"].values[0]} is at or below {aperture_value}')
            aperture_upper_candidates = df_sorted[df_sorted["ApertureValue"] >= aperture_value]
            for one_candidate in aperture_upper_candidates['ApertureValue']: 
                #print(one_candidate)
                pass
            aperture_upper_candidates = aperture_upper_candidates[aperture_upper_candidates[param].notna()]
            aperture_upper = aperture_upper_candidates.head(1)
            #print(f'aperture upper bound of {aperture_upper["ApertureValue"].values[0]} is above {aperture_value}')

            # Check if bounds are valid
            if focal_lower.empty or focal_upper.empty or focus_lower.empty or focus_upper.empty or aperture_lower.empty or aperture_upper.empty:
                print(f"Interpolation not possible for parameter {param} due to missing bounds.")
                interpolated[param] = None
                continue

            # Extract values for interpolation
            try:
                c000 = float(focal_lower.iloc[0][param])
                c001 = float(aperture_upper.iloc[0][param])
                c010 = float(focus_upper.iloc[0][param])
                c011 = float(focus_upper.iloc[0][param])
                c100 = float(focal_upper.iloc[0][param])
                c101 = float(aperture_upper.iloc[0][param])
                c110 = float(focus_upper.iloc[0][param])
                c111 = float(aperture_upper.iloc[0][param])

                '''
                print(f"Surrounding points for {param}:")
                print("c000:", c000, "c001:", c001, "c010:", c010, "c011:", c011, "c100:", c100, "c101:", c101, "c110:", c110, "c111:", c111)

                '''
            except (KeyError, ValueError, TypeError) as e:
                print(f"Error processing parameter {param}: {e}")
                interpolated[param] = None
                continue

            # Perform trilinear interpolation
            try:
                interpolated[param] = trilinear_interpolate(
                    float(focal_lower["FocalLength"].values[0]), float(focal_upper["FocalLength"].values[0]),
                    float(focus_lower["FocusDistance"].values[0]), float(focus_upper["FocusDistance"].values[0]),
                    float(aperture_lower["ApertureValue"].values[0]), float(aperture_upper["ApertureValue"].values[0]),
                    c000, c001, c010, c011, c100, c101, c110, c111,
                    focal_length, focus_distance, aperture_value
                )
            except ZeroDivisionError as e:
                print(f"Division by zero encountered during interpolation for parameter {param}: {e}")
                interpolated[param] = None

        return interpolated

    def get_available_data(self, df):
        """
        Return available data for TCA, vignetting, and distortion.

        Args:
            df (pd.DataFrame): DataFrame of candidate profiles.

        Returns:
            dict: Dictionary of available parameter values.
        """
        available_data = {}
        params = [
            "LensPrettyName",
            "ImageXCenter", "ImageYCenter", "FocalLengthX", "FocalLengthY",
            "RadialDistortParam1", "RadialDistortParam2", "RadialDistortParam3",
            "VignetteModelParam1", "VignetteModelParam2", "VignetteModelParam3",
            "TCA_RedGreen_Radial1", "TCA_RedGreen_Radial2", "TCA_RedGreen_Radial3",
            "TCA_Green_Radial1", "TCA_Green_Radial2", "TCA_Green_Radial3",
            "TCA_BlueGreen_Radial1", "TCA_BlueGreen_Radial2", "TCA_BlueGreen_Radial3"
        ]
        for param in params:
            if param in df.columns:
                value = df[param].dropna().unique().tolist()
                if value != []:
                    available_data[param] = value[0]
        return available_data

    def find_lens_profile(self, cam_maker, cam_model, lens_model, focal_length, focus_distance, aperture_value, interpolate=True):
        """
        Find or interpolate lens profile data for a given camera/lens/focal/aperture/distance.

        Args:
            cam_maker (str): Camera make.
            cam_model (str): Camera model.
            lens_model (str): Lens model.
            focal_length (float): Focal length.
            focus_distance (float): Focus distance.
            aperture_value (float): Aperture value.
            interpolate (bool): Whether to interpolate if no exact match.

        Returns:
            dict or None: Dictionary of profile data, or None if not found.
        """
        # 1. Score and lock the best combo
        lens_dict = {
            'cam_maker': cam_maker,
            'cam_model': cam_model,
            'lens_model': lens_model,
            'focal_length': focal_length,
            'distance': focus_distance,
            'aperture': aperture_value
        }
        scored = score_lens_profile(lens_dict, self)
        if not scored or scored[0]['score'] == 0:
            print('No good profile found')
            return None

        '''
        for _ in range(10):
            print(f'scored {_} is {scored[_]}')
        '''
        best_profile = scored[0]['profile']
        locked_make = best_profile['Make']
        if best_profile['Model'] != ''and pd.notna(best_profile['Model']):
        
            locked_model = best_profile['Model']
        else:
            locked_model = ''
        locked_lens = best_profile['Lens']

        # 2. Gather all profiles for the locked combo
        if locked_model != '':
            print(f'locked model is {locked_model}')
        
            df = self.data[
                (self.data['Make'].str.lower() == locked_make.lower()) &
                (self.data['Model'].str.lower() == locked_model.lower()) &
                (self.data['Lens'].str.lower() == locked_lens.lower())
            ]
        else:
            df = self.data[
                (self.data['Make'].str.lower() == locked_make.lower()) &
                (self.data['Lens'].str.lower() == locked_lens.lower())
            ]

        if df.empty:
            print('No profiles found for locked combo')
            return None

        # 3. Check for an exact match
        exact_match = self.get_exact_match(df, focal_length, focus_distance, aperture_value)
        if not exact_match.empty:
            print('Returning exact match')
            return {
                "Make": locked_make,
                "Model": locked_model,
                "Lens": locked_lens,
                "FocalLength": focal_length,
                "FocusDistance": focus_distance,
                "ApertureValue": aperture_value,
                **self.get_available_data(exact_match)
            }

        # 4. Try to interpolate
        if interpolate:
            interpolated = self.interpolate_values(df, focal_length, focus_distance, aperture_value)
            if interpolated and any(v is not None for v in interpolated.values()):
                print('Returning interpolated values')
                return {
                    "Make": locked_make,
                    "Model": locked_model,
                    "Lens": locked_lens,
                    "FocalLength": focal_length,
                    "FocusDistance": focus_distance,
                    "ApertureValue": aperture_value,
                    **interpolated
                }

        # 5. If interpolation fails, pick the nearest profile
        print('Returning nearest profile')
        df = df.copy()
        df['distance'] = ((df['FocalLength'] - focal_length)**2 +
                        (df['FocusDistance'] - focus_distance)**2 +
                        (df['ApertureValue'] - aperture_value)**2) ** 0.5
        nearest = df.loc[df['distance'].idxmin()]

        available_data = self.get_available_data(df[df['distance'] == nearest['distance']])

        return {
            "Make": locked_make,
            "Model": locked_model,
            "Lens": locked_lens,
            "FocalLength": float(nearest['FocalLength']), #attempting to get away from numpy array returns
            "FocusDistance": float(nearest['FocusDistance']),
            "ApertureValue": float(nearest['ApertureValue']),
            **available_data
        }

def score_vignette_profiles(lens_dict, df):
    """
    Score profiles with exact make/model/lens match, requiring vignette data.
    Higher score for higher aperture and closer focal length.

    Args:
        lens_dict (dict): Dictionary with keys 'focal_length' and 'aperture'.
        df (pd.DataFrame): DataFrame filtered for exact make/model/lens.

    Returns:
        list: List of dicts with keys 'profile_idx', 'score', 'reasons', 'profile', sorted by score descending.
    """
    focal_length = lens_dict.get('focal_length')
    aperture = lens_dict.get('aperture')
    vignette_params = [
        "VignetteModelParam1", "VignetteModelParam2", "VignetteModelParam3"
    ]
    scored = []
    for idx, profile in df.iterrows():
        # Require at least one vignette parameter to be present and not NaN
        has_vignette = any(
            (param in profile and pd.notna(profile[param])) for param in vignette_params
        )
        if not has_vignette:
            continue  # Skip profiles without vignette

        score = 0
        reasons = []

        # Aperture (higher is better)
        if isinstance(profile['ApertureValue'], float) and isinstance(aperture, float):
            score += profile['ApertureValue'] * 10
            reasons.append(f'Aperture: {profile["ApertureValue"]}')
        else:
            reasons.append('No aperture value')

        # Focal length closeness (higher score for closer)
        if isinstance(profile['FocalLength'], float) and isinstance(focal_length, float):
            focal_diff = abs(profile['FocalLength'] - focal_length)
            score += max(0, 100 - focal_diff * 100)
            reasons.append(f'Focal length diff: {focal_diff:.3f}')
        else:
            reasons.append('No focal length')

        scored.append({
            'profile_idx': idx,
            'score': score,
            'reasons': reasons,
            'profile': profile
        })

    return sorted(scored, key=lambda x: x['score'], reverse=True)

def score_tca_profiles(lens_dict, df):
    """
    Score profiles with exact make/model/lens match, requiring TCA data.
    Higher score for closer focal length, and higher focus distance.

    Args:
        df (pd.DataFrame): DataFrame filtered for exact make/model/lens.
        lens_dict (dict): Dictionary with keys 'focal_length' and 'distance'.

    Returns:
        list: List of dicts with keys 'profile_idx', 'score', 'reasons', 'profile', sorted by score descending.
    """
    focal_length = lens_dict.get('focal_length')
    focus_distance = lens_dict.get('distance')
    tca_params = [
        "TCA_RedGreen_Radial1", "TCA_RedGreen_Radial2", "TCA_RedGreen_Radial3",
        "TCA_Green_Radial1", "TCA_Green_Radial2", "TCA_Green_Radial3",
        "TCA_BlueGreen_Radial1", "TCA_BlueGreen_Radial2", "TCA_BlueGreen_Radial3"
    ]
    scored = []
    for idx, profile in df.iterrows():
        # Require at least one TCA parameter to be present and not NaN
        has_tca = any(
            (param in profile and pd.notna(profile[param])) for param in tca_params
        )
        if not has_tca:
            continue  # Skip profiles without TCA

        score = 0
        reasons = []

        # Focal length closeness (higher score for closer)
        if isinstance(profile['FocalLength'], float) and isinstance(focal_length, float):
            focal_diff = abs(profile['FocalLength'] - focal_length)
            score += max(0, 100 - focal_diff * 100)
            reasons.append(f'Focal length diff: {focal_diff:.3f}')
        else:
            reasons.append('No focal length')

        # Focus distance (higher is better)
        if isinstance(profile['FocusDistance'], float):
            score += profile['FocusDistance'] * 10
            reasons.append(f'Focus distance: {profile["FocusDistance"]}')
        else:
            reasons.append('No focus distance')

        scored.append({
            'profile_idx': idx,
            'score': score,
            'reasons': reasons,
            'profile': profile
        })

    return sorted(scored, key=lambda x: x['score'], reverse=True)

def filter_profiles_by_best_combo(scored_profiles, db):
    """
    Given a scored profile list, return all profiles from df that match the top make/model/lens combo.

    Args:
        scored_profiles (list): List of dicts from score_lens_profile, sorted by score descending.
        df (pd.DataFrame): The full DataFrame of profiles.

    Returns:
        pd.DataFrame: DataFrame filtered for the top make/model/lens combo.
    """
    if not scored_profiles:
        return pd.DataFrame()  # Empty DataFrame if no profiles

    df = db.data
    top_profile = scored_profiles[0]['profile']
    make = top_profile['Make']
    model = top_profile['Model']
    lens = top_profile['Lens']

    filtered = df[
        (df['Make'].str.lower() == str(make).lower()) &
        (df['Model'].str.lower() == str(model).lower()) &
        (df['Lens'].str.lower() == str(lens).lower())
    ]
    return filtered

def score_lens_profile(lens_dict, db):
    """
    Score all profiles in the database for similarity to the requested lens/camera/focal/aperture/distance.

    Args:
        lens_dict (dict): Dictionary with keys 'cam_maker', 'cam_model', 'lens_model', 'focal_length', 'distance', 'aperture'.
        db (LensProfileDatabase): The lens profile database.

    Returns:
        list: List of dicts with keys 'profile_idx', 'score', 'reasons', 'profile', sorted by score descending.
    """
    scores = []
    for idx, profile in db.data.iterrows():
        score = 0
        reasons = []
        
        if str(lens_dict['cam_maker']).lower() == str(profile['Make']).lower():

            if isinstance(profile['Model'], str):
                model_similarity = ratio(lens_dict['cam_model'].lower(), profile['Model'].lower())
                #print(f'model similarity is {model_similarity}')
                #print('3')
                if model_similarity > 99.0:
                    score += 18
                    reasons.append('High Camera model match')
                elif model_similarity > 75.0:
                    score += 2
                    reasons.append('Moderate Camera model match')
            else:
                #print(f'somehow model {profile["Model"]} is not a string')
                pass

            if isinstance(profile['Lens'], str):
                lens_similarity = ratio(lens_dict['lens_model'].lower(), profile['Lens'].lower())
                #print('5')
                if lens_similarity > 99.0:
                    score += 35
                    reasons.append(f'High Lens model match of {lens_dict["lens_model"]} to {profile["Lens"]}')
                elif lens_similarity > 75.0:
                    score += 10
                    reasons.append('Moderate Lens model match')

            if isinstance(profile['FocalLength'], float) and isinstance(lens_dict['focal_length'], float):
                #print(f'profile focal length type is {type(profile["FocalLength"])}')
                #print(f'lens focal length type is {type(lens_dict["focal_length"])}')
                focal_length_diff = abs(lens_dict['focal_length'] - profile['FocalLength'])
                #print('7')
                if focal_length_diff < 0.1:
                    score += 15
                    reasons.append(f'Focal length match of {lens_dict["focal_length"]} to {profile["FocalLength"]}')
                elif focal_length_diff < 0.5:
                    score += 5
                    reasons.append(f'Focal length close match of {lens_dict["focal_length"]} to {profile["FocalLength"]}')
                else:
                    #print(f'lens dict focal length of {lens_dict["focal_length"]} and profile focal length of {profile["FocalLength"]} are too far apart')
                    pass
            
            if isinstance(profile['FocusDistance'], float) and isinstance(lens_dict['distance'], float):
                focus_distance_diff = abs(lens_dict['distance'] - profile['FocusDistance'])
                #print('9')
                if focus_distance_diff < 0.1:
                    score += 15
                    reasons.append(f'Focus distance match of {lens_dict["distance"]} to {profile["FocusDistance"]}')
                elif focus_distance_diff < 0.5:
                    score += 5
                    reasons.append(f'Focus distance close match of {lens_dict["distance"]} to {profile["FocusDistance"]}')
                else:
                    #print(f'lens dict focus distance of {lens_dict["distance"]} and profile focus distance of {profile["FocusDistance"]} are too far apart')
                    pass
     

            if isinstance(profile['ApertureValue'], float) and isinstance(lens_dict['aperture'], float):
                aperture_diff = abs(lens_dict['aperture'] - profile['ApertureValue'])
                #print('11')
                if aperture_diff < 0.1:
                    score += 15
                    reasons.append(f'Aperture match of {lens_dict["aperture"]} to {profile["ApertureValue"]}')
                elif aperture_diff < 0.5:
                    score += 5
                    reasons.append(f'Aperture close match of {lens_dict["aperture"]} to {profile["ApertureValue"]}')
                else:
                    #print(f'lens dict aperture of {lens_dict["aperture"]} and profile aperture of {profile["ApertureValue"]} are too far apart')
                    pass

            scores.append({
                'profile_idx': idx,
                'score': score,
                'reasons': reasons,
                'profile': profile
            })

    return sorted(scores, key=lambda x: x['score'], reverse=True)

def demo():
    """
    Demonstration function for LensProfileDatabase usage.
    Loads the database and prints example profile queries.
    """
    db = LensProfileDatabase(lcp_directory=cfg.LCP_DIR,
                             pickle_file=cfg.PICKLE_FILE,
                             force_reload=False)

    # profile = db.find_lens_profile(make="Canon",
    #                                model="Canon EOS R",
    #                                lens="RF28-70mm F2 L USM",
    #                                focal_length=28.0,
    #                                focus_distance=10000.0,
    #                                aperture_value=6.0,
    #                                interpolate=False)
    
    
    # profile2 = db.find_lens_profile(make='Canon',
    #                                model='Canon EOS 5D Mark II',
    #                                lens='EF24-70mm f/2.8L II USM',
    #                                focal_length=24.0,
    #                                focus_distance=0.37,
    #                                aperture_value=6.918863,
    #                                interpolate=False)
    
    # profile3 = db.find_lens_profile(make='Canon',
    #                                model='Canon EOS 5D Mark II',
    #                                lens='EF24-70mm f/2.8L II USM',
    #                                focal_length=24.0,
    #                                focus_distance=0.69,
    #                                aperture_value=6.918863,
    #                                interpolate=False)

    profile4 = db.find_lens_profile(cam_maker='Canon',
                                   cam_model='Canon EOS 5D Mark II',
                                   lens_model='EF24-70mm f/2.8L II USM',
                                   focal_length=24.0,
                                   focus_distance=0.45,
                                   aperture_value=6.918863,
                                   interpolate=True)
    
    profile5 = db.find_lens_profile(cam_maker='NIKON CORPORATION',
                                   cam_model='NIKON D3X',
                                   lens_model='24.0 mm f/2.8',
                                   focal_length=24.0,
                                   focus_distance=0.3,
                                   aperture_value=2.970854,
                                   interpolate=False)

    profile6 = db.find_lens_profile(cam_maker='SIGMA',
                                   cam_model='SIGMA',
                                   lens_model='24.0-105.0 mm',
                                   focal_length=24.0,
                                   focus_distance=10000.0,
                                   aperture_value=24.0,
                                   interpolate=False)
    
    #print(profile)
    #print(profile2)
    #print(profile3)
    print(profile4)
    print(profile5)
    print(profile6)

if __name__ == '__main__':
    print('Run "demo" to test the LensProfileDatabase class.')
    demo()