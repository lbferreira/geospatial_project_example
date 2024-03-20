"""Provides a class to scan Sentinel-2 L2A (SR) data in GCER data server."""

import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


import pandas as pd

from . import metadata_extraction as me


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
        # Remove possible None values (tiles without data under the specified conditions)
        dataset_info = list(filter(lambda x: x is not None, dataset_info))
        if len(dataset_info) == 0:
            raise Exception("No data were found under the specified conditions.")
        dataset_info = pd.concat(dataset_info, axis=0).reset_index(drop=True)
        dataset_info = dataset_info.sort_values(by=["tile", "datetime"]).reset_index(drop=True)
        self._check_for_duplicates(dataset_info)
        return dataset_info

    # -----------------------#
    # --- Private methods ---#
    # -----------------------#

    def _scan_tile(
        self, tile_path: Path, resolution: int, bands: List[str]
    ) -> Optional[pd.DataFrame]:
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

        with ThreadPoolExecutor(max_workers=32) as executor:
            dataset_info = executor.map(
                lambda date_folder, date: self._scan_date(date_folder, date, resolution, bands),
                dates_paths,
                dates,
            )
        # Remove None values, wich can be returned when the cloud cover is over the limit
        dataset_info = list(filter(lambda x: x is not None, dataset_info))
        if len(dataset_info) == 0:
            return None
        dataset_info = pd.concat(dataset_info, axis=0).reset_index(drop=True)
        dataset_info.insert(0, "tile", tile_path.name)
        dataset_info.insert(1, "resolution", resolution)
        return dataset_info

    # Iterate over the dates
    def _scan_date(
        self, date_folder: Path, date: datetime, resolution: int, bands: List[str]
    ) -> Optional[pd.DataFrame]:
        """Scan a date and get the files paths and metadata (DataFrame) for a given resolution and bands."""
        safe_folder = self._get_next_level_folders(date_folder)
        if len(safe_folder) > 1:
            msg = f"More than one SAFE folder were found in the folder {str(date_folder)}. Using the first one."
            warnings.warn(msg)
        safe_folder = safe_folder[0]

        # Get metadata
        metatada_file = safe_folder / self.METADATA_FILE_NAME
        raw_metadata = me.get_raw_metadata(metatada_file)

        # Safety check: date in the metadata file must match the date in the folder
        if raw_metadata["datetime"].date() != date.date():
            raise Exception(
                f"Date mismatch. The date in the metadata file is {raw_metadata['datetime'].date()}, but the date in the folder is {date.date()}"
            )

        # Filter by cloud cover
        if raw_metadata["cloud_coverage"] > self.max_cloud_cover:
            return None

        granule_folder = safe_folder / "GRANULE"
        bands_info = self._get_images_paths(granule_folder, resolution, bands)
        formatted_metadata = me.get_formatted_metadata(raw_metadata, bands_info)
        return formatted_metadata

    def _get_images_paths(
        self, granule_folder: Path, resolution: int, bands: List[str]
    ) -> Dict[str, Path]:
        """A dictionary with the band names as keys and the file paths as values is returned."""
        next_level_folder = self._get_next_level_folders(granule_folder)
        if len(next_level_folder) > 1:
            raise Exception(
                f"More than one folder were found inside the GRANULE folder. Path {str(granule_folder)}"
            )
        folder = next_level_folder[0]
        folder = folder / f"IMG_DATA/{self.RESOLUTION_FOLDERS[resolution]}/"
        bands_paths = list(folder.glob("*.jp2"))
        # Keep only the bands requested
        bands_paths = [p for p in bands_paths if p.name.split("_")[-2] in bands]
        band_names = [p.name.split("_")[-2] for p in bands_paths]
        return {k: v for k, v in zip(band_names, bands_paths)}

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
