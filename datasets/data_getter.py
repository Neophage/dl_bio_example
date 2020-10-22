from . import ds_natural_images
from . import ds_immunfix_images


def get_data_loaders(dataset, batch_size, **kwargs):
    if dataset == 'nat_im':
        return {'train': ds_natural_images.get_dataloader(
            True, batch_size, kwargs['split_index']),
            'val': ds_natural_images.get_dataloader(
            False, batch_size, kwargs['split_index']),
            'test': None
        }
    elif dataset == 'imfx_im':
        return {'train': ds_immunfix_images.get_dataloader(
            True, batch_size, kwargs['split_index']),
            'val': ds_immunfix_images.get_dataloader(
            False, batch_size, kwargs['split_index']),
            'test': None
        }
        
def get_test_dataloader(dataset, subset, split_index=0):
    if dataset == 'imfx_im':
        return ds_immunfix_images.get_test_dataloader(subset, split_index)
		
def get_class_names(dataset):
	if dataset == 'nat_im':
		return ds_natural_images.CLASSES
		
	elif dataset == 'imfx_im':
		return ds_immunfix_images.CLASSES