"""Provides a class to scan Sentinel-2 L2A (SR) data in GCER data server."""
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import rioxarray
import xarray


class S2L2AScanner:
    # Parameters for Sentinel-2 L2A data in GCER data server
    METADATA_FILE_NAME = "MTD_MSIL2A.xml"
    VALID_RESOLUTIONS = [10, 20, 60]
    BANDS_PER_RESOLUTION = {
        10: ["B02", "B03", "B04", "B08", "AOT", "TCI", "WVP"],
        20: [
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B11",
            "B12",
            "SCL",
            "AOT",
            "TCI",
            "WVP",
        ],
        60: [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B8A",
            "B09",
            "B11",
            "B12",
            "SCL",
            "AOT",
            "TCI",
            "WVP",
        ],
    }
    RESOLUTION_FOLDERS = {
        10: "R10m",
        20: "R20m",
        60: "R60m",
    }

    def __init__(
        self,
        root_folder: str,
        ini_date: str,
        end_date: str,
        max_cloud_cover: int,
        tiles: Optional[List[str]] = None,
    ):
        """Scans Sentinel-2 L2A (SR) folder and get the files and metadata based.

        Args:
            root_folder (str): path to the Sentinel-2 L2A (SR) folder in GCER data server. Something like: "Z:/dbcenter/images/sentinel/scenes/level_sr/"
            ini_date (str): initial date in the format "YYYY-MM-DD"
            end_date (str): final date in the format "YYYY-MM-DD"
            max_cloud_cover (int): maximum cloud cover allowed. Between 0 and 100.
            tiles (Optional[List[str]], optional): list of tiles to scan. If None, all tiles will be scanned. Defaults to None.
        """
        self.root_folder = Path(root_folder)
        self.ini_date = datetime.strptime(ini_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.max_cloud_cover = max_cloud_cover
        self.tiles = tiles

    def get_available_tiles(self) -> List[str]:
        """Get the available tiles from the root folder."""
        return [p.name for p in self._get_next_level_folders(self.root_folder)]

    def get_available_dates_per_tile(self, tile: str) -> List[datetime]:
        """Get all available dates for a tile."""
        tile_path = self.root_folder / tile
        dates_paths = self._get_next_level_folders(tile_path)
        dates = [
            datetime(year=int(p.name[0:4]), month=int(p.name[4:6]), day=int(p.name[6:8]))
            for p in dates_paths
        ]
        return dates

    # TODO: implement it
    # def get_intersecting_tiles(self, geometry) -> List[str]:
    #     """Get the tiles that intersect with the provided geometry."""
    #     pass

    def get_files_info(self, resolution: int, bands: List[str]) -> pd.DataFrame:
        """Get the file paths and metadata for a given resolution and bands.

        Args:
            resolution (int): resolution in meters. Valid values are 10, 20 and 60.
            bands (List[str]): list of bands to get. Each resolution has a different set of valid bands.

        Returns:
            pd.DataFrame: dataframe with the metadata and file paths.
        """
        # Arguments validation
        self._check_resolution(resolution)
        self._check_bands(bands, resolution)

        tiles_found = self._get_tile_paths()

        # Call the function to scan each tile using future multithreading
        with ThreadPoolExecutor(max_workers=8) as executor:
            dataset_info = executor.map(
                lambda x: self._scan_tile(x, resolution, bands), tiles_found
            )
        # Remove possible empty lists
        dataset_info = list(filter(lambda x: len(x) > 0, dataset_info))
        dataset_info_flat = []
        for dataset_info_tile in dataset_info:
            dataset_info_flat.extend(dataset_info_tile)

        if len(dataset_info_flat) == 0:
            raise Exception("No data were found under the specified conditions.")

        dataset_info = pd.concat(dataset_info_flat, axis=0).reset_index(drop=True)
        dataset_info = dataset_info.sort_values(by=["tile", "datetime"]).reset_index(drop=True)
        self._check_for_duplicates(dataset_info)
        return dataset_info

    # -----------------------#
    # --- Private methods ---#
    # -----------------------#

    def _scan_tile(self, tile_path: Path, resolution: int, bands: List[str]) -> List[pd.DataFrame]:
        """Scan a tile and get the files and metadata for a given resolution and bands.
        Multithreading is used to scan the dates."""
        # Date-level folders
        dates_paths = self._get_next_level_folders(tile_path)
        dates = [
            datetime(year=int(p.name[0:4]), month=int(p.name[4:6]), day=int(p.name[6:8]))
            for p in dates_paths
        ]
        # Filter by date
        for i in range(len(dates)):
            if dates[i] < self.ini_date or dates[i] > self.end_date:
                dates[i] = None
                dates_paths[i] = None
        dates = [d for d in dates if d is not None]
        dates_paths = [p for p in dates_paths if p is not None]

        # Iterate over the dates
        def scan_date(date_folder: Path, date: datetime) -> Optional[pd.DataFrame]:
            safe_folder = self._get_next_level_folders(date_folder)
            if len(safe_folder) > 1:
                # raise Exception(f"More than one SAFE folder were found in the folder {str(p)}")
                msg = f"More than one SAFE folder were found in the folder {str(date_folder)}. Using the first one."
                warnings.warn(msg)
            safe_folder = safe_folder[0]

            # Get metadata
            metatada_file = safe_folder / self.METADATA_FILE_NAME
            metadata = self._get_metadata(metatada_file)

            # Safety check: date in the metadata file must match the date in the folder
            if metadata["datetime"].date() != date.date():
                raise Exception(
                    f"Date mismatch. The date in the metadata file is {metadata['datetime'].date()}, but the date in the folder is {date.date()}"
                )

            # Filter by cloud cover
            if metadata["cloud_coverage"] > self.max_cloud_cover:
                return None

            # Add tile and resolution as metadata
            metadata["tile"] = tile_path.name
            metadata["resolution"] = resolution

            # Get file path
            image_path = safe_folder / "GRANULE"
            next_level_folder = self._get_next_level_folders(image_path)
            if len(next_level_folder) > 1:
                raise Exception(
                    f"More than one folder were found inside the GRANULE folder. Path {str(image_path)}"
                )
            image_path = next_level_folder[0]
            image_path = image_path / f"IMG_DATA/{self.RESOLUTION_FOLDERS[resolution]}/"
            bands_paths = list(image_path.glob("*.jp2"))

            # Keep only the bands requested
            bands_paths = [p for p in bands_paths if p.name.split("_")[-2] in bands]

            # Expand the metadata to match the number of bands
            nb_bands = len(bands_paths)
            for key in metadata.keys():
                metadata[key] = [metadata[key]] * nb_bands

            # Add the band name and file path to the metadata
            metadata["band"] = [p.name.split("_")[-2] for p in bands_paths]
            metadata["file_path"] = [p.resolve() for p in bands_paths]

            # Keep only the rigth quantification and offset values for each band
            # On file names band names appear as B0X, but on metadata they appear as BX.
            # Thus, we look for both cases for ensuring that the right values are used.
            boa_add_offset = []
            for band in metadata["band"]:
                offset = metadata.get(f"boa_add_offset_{band}", None)
                if offset is None and band.startswith("B0"):
                    band = band.replace("B0", "B")
                    offset = metadata.get(f"boa_add_offset_{band}", None)
                if offset is None:
                    offset = [np.nan]
                boa_add_offset.append(offset[0])

            # Remove previous keys related to boa_add_offset
            for key in list(metadata.keys()):
                if key.startswith("boa_add_offset_"):
                    metadata.pop(key)

            # Create a dataframe with the metadata
            image_dataset = pd.DataFrame(metadata)
            # For organization purposes, the column band_boa_add_offset is inserted before boa_quantification_value
            columns = list(image_dataset.columns)
            idx = columns.index("boa_quantification_value")
            image_dataset.insert(idx, "band_boa_add_offset", boa_add_offset)
            return image_dataset

        with ThreadPoolExecutor(max_workers=32) as executor:
            dataset_info = executor.map(scan_date, dates_paths, dates)
        # Remove None values, wich can be returned when the cloud cover is over the limit
        dataset_info = list(filter(lambda x: x is not None, dataset_info))
        return dataset_info

    def _check_for_duplicates(self, dataset_info: pd.DataFrame) -> None:
        duplicated = dataset_info.duplicated(subset=["product_id", "band"])
        if duplicated.sum() > 0:
            raise Exception(
                "There are duplicated data (The same product_id/band appears more than a single time)."
            )

    def _get_tile_paths(self) -> List[Path]:
        """Get the paths to the tiles found in the root folder and selected by the user."""
        # First level of folders: tile
        tiles_found = self._get_next_level_folders(self.root_folder)
        if self.tiles is not None:
            tiles_found = [p for p in tiles_found if p.name in self.tiles]
            tiles_found_names = [p.name for p in tiles_found]
            tiles_not_found = list(set(self.tiles) - set(tiles_found_names))
            if len(tiles_not_found) > 0:
                warnings.warn(f"Tiles not found: {tiles_not_found}")
        if len(tiles_found) == 0:
            raise Exception("No tiles were found.")
        return tiles_found

    def _get_next_level_folders(self, path: Path) -> List[Path]:
        """Get the next level of folders from a path."""
        return [p for p in path.glob("*") if p.is_dir()]

    def _get_metadata(self, metadata_file_path: str) -> dict:
        """Get the metadata from a Sentinel-2 XML file."""
        metadata = {}
        root = ElementTree.parse(metadata_file_path).getroot()
        # XML namespace prefix
        ns_prefix = root.tag[: root.tag.index("}") + 1]
        ns_dict = {"n1": ns_prefix[1:-1]}
        # General info
        general_info_node = root.find("n1:General_Info", ns_dict)
        geom_info_node = general_info_node.find("Product_Info")
        sensing_time_node = geom_info_node.find("PRODUCT_START_TIME")
        sensing_time_str = sensing_time_node.text.strip()
        metadata["product_id"] = geom_info_node.find("PRODUCT_URI").text.strip()
        metadata["datetime"] = datetime.strptime(sensing_time_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        # Quantification values
        image_charac = general_info_node.find("Product_Image_Characteristics")
        quantification_values = image_charac.find("QUANTIFICATION_VALUES_LIST")
        metadata["boa_quantification_value"] = float(
            quantification_values.find("BOA_QUANTIFICATION_VALUE").text
        )
        metadata["aot_quantification_value"] = float(
            quantification_values.find("AOT_QUANTIFICATION_VALUE").text
        )
        metadata["wvp_quantification_value"] = float(
            quantification_values.find("WVP_QUANTIFICATION_VALUE").text
        )
        # Retrieve band names
        spectral_info = image_charac.find("Spectral_Information_List")
        band_ids = np.arange(0, 13)
        band_name_by_id = {}
        for band_id in band_ids:
            band_name = spectral_info.find(f"Spectral_Information[@bandId='{band_id}']").get(
                "physicalBand"
            )
            band_name_by_id[band_id] = band_name
        # Retrieve band offsets
        # BOA_ADD_OFFSET_VALUES_LIST are available only for baseline processing 04.00 and above
        # Obtaining the offsets for each band is important to properly convert the data to
        # reflectance and to harmonize the data before and after baseline 04.00
        # https://sentinels.copernicus.eu/web/sentinel/-/deployment-of-sentinel-2-processing-baseline-04.00?redirect=%2Fweb%2Fsentinel%2Faccess-to-sentinel-data-via-the-copernicus-data-space-ecosystem
        # https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/processing-baseline
        boa_add_offset_values = image_charac.find("BOA_ADD_OFFSET_VALUES_LIST")
        if boa_add_offset_values is not None:
            for band_id, band_name in band_name_by_id.items():
                boa_add_offset = float(
                    boa_add_offset_values.find(f"BOA_ADD_OFFSET[@band_id='{band_id}']").text
                )
                metadata[f"boa_add_offset_{band_name}"] = boa_add_offset
        else:
            for band_name in band_name_by_id.values():
                metadata[f"boa_add_offset_{band_name}"] = 0
        # Quality indicators
        geom_info_node = root.find("n1:Quality_Indicators_Info", ns_dict)
        metadata["cloud_coverage"] = float(geom_info_node.find("Cloud_Coverage_Assessment").text)
        geocoding_node = geom_info_node.find("Image_Content_QI")
        metadata["nodata_perc"] = float(geocoding_node.find("NODATA_PIXEL_PERCENTAGE").text)
        metadata["cloud_shadow_perc"] = float(geocoding_node.find("CLOUD_SHADOW_PERCENTAGE").text)
        metadata["medium_prob_cc"] = float(
            geocoding_node.find("MEDIUM_PROBA_CLOUDS_PERCENTAGE").text
        )
        metadata["high_prob_cc"] = float(geocoding_node.find("HIGH_PROBA_CLOUDS_PERCENTAGE").text)
        metadata["cirrus_cc"] = float(geocoding_node.find("THIN_CIRRUS_PERCENTAGE").text)
        metadata["snow_perc"] = float(geocoding_node.find("SNOW_ICE_PERCENTAGE").text)
        return metadata

    def _check_resolution(self, resolution: int) -> None:
        """Check if the resolution is valid."""
        if resolution not in self.VALID_RESOLUTIONS:
            raise Exception("The resolution must be 10, 20 or 60 meters.")

    def _check_bands(self, bands: List[str], resolution: int) -> None:
        """Check if the bands are valid."""
        for band in bands:
            if band not in self.BANDS_PER_RESOLUTION[resolution]:
                raise Exception(
                    f"The band {band} is not valid. For resolution {resolution}, the valid bands are {self.BANDS_PER_RESOLUTION[resolution]}."
                )


def load_multiple_tile_dataarray(
    files_info: pd.DataFrame,
    dtype: Optional[np.dtype] = None,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs,
) -> List[xarray.DataArray]:
    """Loads data from multiple tiles as xarray DataArray objects.
    When loading data using chunks, dask is automatically used to parallelize the computation.
    All computations are lazy, meaning that the data is not loaded until it is needed.

    Args:
        files_info (pd.DataFrame): files info dataframe.
        dtype (Optional[np.dtype], optional): a numpy dtype to cast the data to. If None,
        the original dtype is used. Defaults to None.
        chunks (Optional[Dict[str, int]], optional): chunks to use when loading
        the data using dask. If None, no chunks are used. Defaults to None.
        kwargs: keyword arguments to pass to rioxarray.open_rasterio.

    Returns:
        List[xarray.DataArray]: list of DataArray containing the loaded data.
    """
    data_per_tile = []
    for tile, tile_rows in files_info.groupby("tile"):
        data_array = load_tile_dataarray(tile_rows, dtype=dtype, chunks=chunks, **kwargs)
        data_per_tile.append(data_array)
    return data_per_tile


def load_tile_dataarray(
    tile_files_info: pd.DataFrame,
    dtype: Optional[np.dtype] = None,
    chunks: Optional[Dict[str, int]] = None,
    **kwargs,
) -> xarray.DataArray:
    """Loads data of a particular tile as an xarray DataArray object.
    When loading data using chunks, dask is automatically used to parallelize the computation.
    All computations are lazy, meaning that the data is not loaded until it is needed.

    Args:
        files_info (pd.DataFrame): files info dataframe.
        dtype (Optional[np.dtype], optional): a numpy dtype to cast the data to. If None,
        the original dtype is used. Defaults to None.
        chunks (Optional[Dict[str, int]], optional): chunks to use when loading
        the data using dask. If None, no chunks are used. Defaults to None.
        kwargs: keyword arguments to pass to rioxarray.open_rasterio.

    Returns:
        xarray.DataArray: DataArray containing the loaded data.
    """
    nb_tiles = len(tile_files_info["tile"].unique())
    if nb_tiles > 1:
        raise ValueError(f"More than one tile found in the dataframe. Found {nb_tiles} tiles.")

    tile = tile_files_info["tile"].iat[0]

    images_per_tile = []
    for _, image_bands_rows in tile_files_info.groupby("product_id"):
        datetime = image_bands_rows["datetime"].iat[0]
        bands_data = []
        for df_row in image_bands_rows.itertuples():
            xda = rioxarray.open_rasterio(df_row.file_path, chunks=chunks, **kwargs)
            xda = xda.assign_coords(band=[df_row.band])
            bands_data.append(xda)
        bands_concat = xarray.concat(bands_data, dim="band")
        bands_concat["time"] = datetime
        images_per_tile.append(bands_concat)
    images_per_tile = xarray.concat(images_per_tile, dim="time")
    images_per_tile["tile"] = tile
    if dtype is not None:
        images_per_tile = images_per_tile.astype(dtype)
    return images_per_tile


def normalize_s2(
    data: xarray.DataArray,
    metadata: pd.DataFrame,
    harmonize: bool = True,
    dtype: np.dtype = np.float32,
) -> xarray.DataArray:
    """Normalizes Sentinel-2 data to reflectance.
    This functions was specially designed to normalize data by removing the offset introduced in processing baseline 04.00.
    Useful links below:
    https://sentinels.copernicus.eu/web/sentinel/-/deployment-of-sentinel-2-processing-baseline-04.00?redirect=%2Fweb%2Fsentinel%2Faccess-to-sentinel-data-via-the-copernicus-data-space-ecosystem
    https://sentinels.copernicus.eu/web/sentinel/technical-guides/sentinel-2-msi/processing-baseline

    Args:
        data (xarray.DataArray): data to normalize.
        metadata (pd.DataFrame): dataframe with the metadata, containig the offsets and
        quantification values for each band.
        harmonize (bool, optional): whether to normalize data to allways have the same pattern used
        before processing baseline 04.00. If False, using a time series with before and after
        the baseline processing 04.00 is not appropriated. Defaults to True.
        dtype (np.dtype, optional): data type to convert the data to. Defaults to np.float32.

    Returns:
        xarray.DataArray: normalized data.
    """
    original_band_order = data.band.values
    original_time_order = data.time.values
    original_dims_order = data.dims
    data = data.astype(dtype)

    normalized_data = []
    # Normalize optical bands
    all_boa_bands = [f"B{band_nb:02}" for band_nb in range(1, 12 + 1)] + ["B8A"]
    boa_bands = [band for band in data.band.values if band in all_boa_bands]
    for band, metadata_band in metadata.groupby("band"):
        if band not in boa_bands:
            continue
        normalized_band_data = []
        for (offset, quantification_value), df_subset in metadata_band.groupby(
            ["band_boa_add_offset", "boa_quantification_value"]
        ):
            time_values = df_subset["datetime"].values
            # Processing baseline 04.00 instroduced an offset of -1000
            # If harmonize is True, we harmonize new data to be compatible
            # with data generated with previous processing baselines
            if harmonize:
                norm = data.sel(band=band, time=time_values).clip(offset * -1)
                norm += offset
                norm /= quantification_value
            else:
                norm = (data.sel(band=band, time=time_values) + offset) / quantification_value
            normalized_band_data.append(norm)
        normalized_band_data = xarray.concat(normalized_band_data, dim="time")
        normalized_data.append(normalized_band_data)

    # Normalize AOT and WVP
    aot_wvp_bands = ["AOT", "WVP"]
    quantification_value_columns = ["aot_quantification_value", "wvp_quantification_value"]
    for band, quant_col in zip(aot_wvp_bands, quantification_value_columns):
        if band in data.band.values:
            metadata_band = metadata[metadata["band"] == band]
            normalized_band_data = []
            for quantification_value, df_subset in metadata_band.groupby(quant_col):
                time_values = df_subset["datetime"].values
                norm = data.sel(band=band, time=time_values) / quantification_value
                normalized_band_data.append(norm)
            normalized_band_data = xarray.concat(normalized_band_data, dim="time")
            normalized_data.append(normalized_band_data)

    # Include any other band (TCI and SCL) without normalization
    other_bands = list(set(data.band.values.tolist()) - set(all_boa_bands + aot_wvp_bands))
    if len(other_bands) > 0:
        normalized_data.append(data.sel(band=other_bands))

    normalized_data = xarray.concat(normalized_data, dim="band")
    # Only to sort the bands and time to keep the original order
    normalized_data = normalized_data.sel(time=original_time_order, band=original_band_order)
    normalized_data = normalized_data.transpose(*original_dims_order)
    return normalized_data
