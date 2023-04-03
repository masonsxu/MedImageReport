
class Config(dict):
    def version_config(self, version):
        hp = {1: {'n_epoch':50, 'batch':32, 'valid_batch':8, 'n_layer':3},
              }
        self['n_epoch'] = hp[version].get('n_epoch', 10)
        self['n_layer'] = hp[version].get('n_layer', 6)
        self['batch'] = hp[version].get('batch', 8)
        self['valid_batch'] = hp[version].get('valid_batch', 8)
        self['w_g'] = 1

        #请自己造训练测试集
        self['train_file'] = 'data/train.csv'
        self['valid_file'] = 'data/valid.csv'
        self['test_file'] = 'data/test.csv'
    
        self['input_l'] = 150
        self['output_l'] = 80
        self['n_token'] = 2000
        self['sos_id'] = 1
        self['eos_id'] = 2
        self['pad_id'] = 0
        
    def __init__(self, version, seed=0):
        self['lr'] = 3e-5
        self['model_dir'] = './checkpoint/%d'%version
        if seed>0:
            self['model_dir'] += '_%d'%seed
        self['output_dir'] = './outputs/%d'%version
        
        self.version_config(version)
        