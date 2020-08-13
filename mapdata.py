from pathlib import Path
import geopandas
import pickle

jar = Path('./mapdata.pickle')
if not jar.exists():
    root = '/home/nullity/dev/jupyter'
    p3145 = (
        geopandas
        .read_file(f'{root}/postcodes-vic/ll_gda94/sde_shape/whole/VIC/VMADMIN/layer/postcode_polygon.shp')
        .query('POSTCODE == "3145"')
        .to_crs(epsg=3112)
    )
    roads_4000 = (
        geopandas
        .read_file(f'{root}/roads-3145/ll_gda94/shape/postcode_polygon/3145-4000/VMTRANS/TR_ROAD.shp')
        .to_crs(epsg=3112)
    )
    roads_200 = geopandas.clip(roads_4000, p3145.buffer(200))
    roads_0 = geopandas.clip(roads_4000, p3145.buffer(0))

    with jar.open(mode='bw') as dest:
        pickle.dump({
            'roads_200': roads_200,
            'roads_0': roads_0,
        }, dest)

else:

    with jar.open(mode='rb') as src:
        locals().update(pickle.load(src))

