import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def construct_thumbnail(storm):
    
    rep_time = storm.sel(time=storm.sum(dim=['lat', 'lon']).idxmax())
    fig, ax = plt.subplots(1)
    ax.imshow(rep_time.to_numpy());
    ax.invert_yaxis()
    ax.axis('off');
    fig.set_size_inches((0.5,0.5))
    plt.close()

    # following strategy taken from the followign stackexchange
    # https://stackoverflow.com/questions/47038538/insert-matplotlib-images-into-a-pandas-dataframe
    figfile = BytesIO()
    fig.savefig(figfile, format='png', pad_inches=0, bbox_inches='tight')
    figfile.seek(0)
    figdata_png = base64.b64encode(figfile.getvalue()).decode()
    imgstr = '<img src="data:image/png;base64,{}" />'.format(figdata_png)

    return imgstr


def display_catalog(catalog_df, nrows=None):
    if nrows:
        return catalog_df.head(nrows).style.format({'data_array': lambda x: construct_thumbnail(x)}).format_index(precision=0)
    else:
        return catalog_df.style.format({'data_array': construct_thumbnail}).format_index(precision=0)

