from pathlib import Path
from typing import List, Optional
import pandas as pd


class ImageScanner:
    def __init__(
        self,
        search_folder: str,
    ) -> None:
        self.search_folder = Path(search_folder)
        self._scan_results_full = None

    def scan(
        self, ini_month: Optional[int] = None, final_month: Optional[int] = None
    ) -> pd.DataFrame:
        # If scan results are already available, just use them. Otherwise, create them.
        # It is useful to store the full scan results to allow subsequent filtering without
        # the need to scan the images again.
        if self._scan_results_full == None:
            self._scan_results_full = self._create_scan_df()

        scan_results = self._filter_scam_results(
            self._scan_results_full, ini_month, final_month
        )
        # Sort by month and band for easier visualization
        scan_results = scan_results.sort_values(by=["month", "band"])
        # Reset index to make index continuous after filtering and sorting
        scan_results = scan_results.reset_index(drop=True)
        return scan_results

    def _filter_scam_results(
        self,
        scan_results: pd.DataFrame,
        ini_month: Optional[int] = None,
        final_month: Optional[int] = None,
    ) -> pd.DataFrame:
        if ini_month is not None:
            scan_results = scan_results[scan_results["month"] >= ini_month]
        if final_month is not None:
            scan_results = scan_results[scan_results["month"] <= final_month]
        return scan_results

    def _create_scan_df(self) -> pd.DataFrame:
        file_paths = self._list_tif_files()
        metadata = self._extract_metadata(file_paths)
        metadata["file_path"] = file_paths
        return pd.DataFrame(metadata)

    def _list_tif_files(self) -> List[Path]:
        return list(self.search_folder.glob("**/*.tif"))

    def _extract_metadata(self, file_paths: List[Path]) -> dict:
        months = [int(p.stem.split("_")[1]) for p in file_paths]
        bands = [p.stem.split("_")[3] for p in file_paths]
        return {"month": months, "band": bands}