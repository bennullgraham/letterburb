from pathlib import Path
import geopandas
import pickle

def get_mapdata(postcode: int) -> dict:
    assert postcode >= 3000 and postcode < 4000

    root = Path('./')
    jar = root / f'./mapdata-{postcode}.pickle'

    if not jar.exists():
        print(f'No cached data for {postcode}. This will take a minute.')
        postcode_path = f'{root}/postcodes-vic/ll_gda94/sde_shape/whole/VIC/VMADMIN/layer/postcode_polygon.shp'
        print(f'Reading postcode boundaries from {postcode_path}')
        postcode_boundary = (
            geopandas
            .read_file(postcode_path)
            .query(f'POSTCODE == "{postcode}"')
            .to_crs(epsg=3112)
        )

        # TODO probably quicker to convert reference system *after* clipping to
        # postcode...
        roads_path = f'{root}/roads-vic/ll_gda2020/shape/whole_of_dataset/vic/VMTRANS/TR_ROAD.shp'
        print(f'Reading roads from {roads_path}')
        roads_vic = (
            geopandas
            .read_file(roads_path)
            .to_crs(epsg=3112)
        )
        roads_buffer = geopandas.clip(roads_vic, postcode_boundary.buffer(500))
        roads_0 = geopandas.clip(roads_buffer, postcode_boundary.buffer(0))

        with jar.open(mode='bw') as dest:
            mapdata = {
                'postcode': postcode_boundary,
                'roads_buffer': roads_buffer,
                'roads_0': roads_0,
            }
            pickle.dump(mapdata,dest)

    else:
        with jar.open(mode='rb') as src:
            mapdata = pickle.load(src)

    return mapdata

