from datetime import datetime
from pathlib import Path
from typing import Dict, List
from xml.etree import ElementTree

import numpy as np
import pandas as pd


def get_raw_metadata(metadata_file_path: str) -> dict:
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
    metadata["medium_prob_cc"] = float(geocoding_node.find("MEDIUM_PROBA_CLOUDS_PERCENTAGE").text)
    metadata["high_prob_cc"] = float(geocoding_node.find("HIGH_PROBA_CLOUDS_PERCENTAGE").text)
    metadata["cirrus_cc"] = float(geocoding_node.find("THIN_CIRRUS_PERCENTAGE").text)
    metadata["snow_perc"] = float(geocoding_node.find("SNOW_ICE_PERCENTAGE").text)
    return metadata


def get_formatted_metadata(raw_metadata: dict, bands: Dict[str, Path]) -> pd.DataFrame:
    """Retuns metadata in a DataFrame anf only for the bands provided."""
    band_names = list(bands.keys())
    band_paths = list(bands.values())
    boa_add_offset_per_band = _get_bands_boa_add_offset(raw_metadata, band_names)
    raw_metadata = _filter_boa_add_offset_keys(raw_metadata)

    # Expand metadata based on the number of bands to prepare it to generate a DataFrame
    nb_bands = len(band_names)
    for key in raw_metadata.keys():
        raw_metadata[key] = [raw_metadata[key]] * nb_bands
    # Generate the DataFrame
    metadata = pd.DataFrame(raw_metadata)

    # Add the band name and file path to the metadata
    metadata["band"] = band_names
    metadata["file_path"] = [p.resolve() for p in band_paths]

    # For organization purposes, the column band_boa_add_offset is inserted before boa_quantification_value
    columns = list(metadata.columns)
    idx = columns.index("boa_quantification_value")
    metadata.insert(idx, "band_boa_add_offset", boa_add_offset_per_band)
    return metadata


######## Private functions ########


def _get_bands_boa_add_offset(metadata: dict, band_names: List[str]) -> List[float]:
    boa_add_offset = []
    for band in band_names:
        offset = metadata.get(f"boa_add_offset_{band}", None)
        if offset is None and band.startswith("B0"):
            band = band.replace("B0", "B")
            offset = metadata.get(f"boa_add_offset_{band}", None)
        if offset is None:
            offset = np.nan
        boa_add_offset.append(offset)
    return boa_add_offset


def _filter_boa_add_offset_keys(metadata: dict) -> dict:
    # Remove previous keys related to boa_add_offset
    selected_keys = [key for key in metadata if not key.startswith("boa_add_offset_")]
    filtered_dict = {key: metadata[key] for key in selected_keys}
    return filtered_dict
