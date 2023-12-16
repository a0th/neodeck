from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydeck as pdk
import geopandas
import json


def h3_hexagon(
    data: pd.DataFrame,
    h3_index: str,
    color: str,
    height: str,
    height_multiplier: int = 10000,
    opacity_column: Optional[str] = None,
    opacity_scalar: Optional[float] = None,
    pickable: bool = False,
    wireframe: bool = False,
    colormap: Optional[str] = None,
):
    """
    Generate a hexagon layer using H3 hexagon IDs.

    Parameters:
    -----------
    data : pd.DataFrame
        The data to be used for the hexagon layer.
    h3_index : str
        The name of the column in `data` containing the H3 hexagon IDs.
    color : str
        The name of the column in `data` containing the fill color for each hexagon.
    height : str
        The name of the column in `data` containing the height for each hexagon.
    pickable : bool, optional
        Whether the hexagons should be pickable, by default False.
    wireframe : bool, optional
        Whether to render the hexagons as wireframes, by default False.

    Returns:
    --------
    pydeck.Layer
        A PyDeck layer representing the hexagon layer.
    """
    # assert columns.
    # keep the column_name*100 parsing?
    # opacity and opacity_scalar are cannot be set at the same time
    assert not (opacity_column is not None and opacity_scalar is not None)

    # if opacity_column is given, it must be in the data
    assert not (opacity_column is not None and opacity_column not in data.columns)

    # if opacity_scalar is given, it must be between 0 and 1
    assert not (opacity_scalar is not None and not 0 <= opacity_scalar <= 1)

    if colormap is not None:
        if colormap not in plt.colormaps():
            raise ValueError(f"colormap {colormap} not found in matplotlib")

    _data = data.copy()

    opacity_series = _get_opacity_series(_data, opacity_column, opacity_scalar)

    _data["__color_column_generated"] = map_series_to_colormap(
        data[color], opacity_series, colormap_name=colormap
    ).tolist()
    if height:
        _data["__height_column_generated"] = _data[height] / _data[height].max()
    return pdk.Layer(
        "H3HexagonLayer",
        _data,
        get_hexagon=h3_index,
        get_fill_color="__color_column_generated",
        get_elevation=f"__height_column_generated*{height_multiplier}",
        auto_highlight=True,
        pickable=pickable,
        extruded=height is not None,
        wireframe=wireframe,
    )


def select_colormap(series):
    if series.dtype == "object":
        if series.nunique() < 10:
            return "tab10"  # qualitative colormap for categorical data
        return "tab20"
    elif np.all(series >= 0):
        return "turbo"  # sequential colormap for non-negative data
    else:
        return "coolwarm"  # diverging colormap for data with both positive and negative values


def _get_opacity_series(data, opacity, opacity_scalar) -> pd.Series:
    if opacity is not None:
        assert opacity in data.columns
        opacity_series = data[opacity]
    elif opacity_scalar is not None:
        opacity_series = pd.Series([int(opacity_scalar * 255)] * len(data))
    else:
        opacity_series = pd.Series([255] * len(data))
    return opacity_series


def map_series_to_colormap(
    series: pd.Series, opacity: pd.Series, colormap_name: Optional[str] = None
):
    """Map a pandas Series to a colormap, returning a Series of RGB values."""

    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if colormap_name is None:
        default_map = select_colormap(series)
        colormap = plt.cm.get_cmap(default_map)
    else:
        colormap = plt.cm.get_cmap(colormap_name)
    if series.dtype == "O":
        series = pd.Series(pd.factorize(series)[0])
    else:
        min_val = series.min()
        max_val = series.max()
        # Normalize the series values to the range [0, 1]
        series = (series - min_val) / (max_val - min_val)

    # Get the RGBA values from the colormap
    rgba_values = colormap(series.values)
    rgb_values = np.array(rgba_values[:, :3] * 255, dtype=int)
    rgb_values_with_opacity = np.concatenate(
        [rgb_values, opacity.to_numpy()[:, np.newaxis]], axis=1
    )
    return rgb_values_with_opacity


def scatter(
    data: pd.DataFrame,
    latitude: str,
    longitude: str,
    fill_color: str,
    radius_scalar: int = 100,
):
    """
    Generate a scatter layer using latitude and longitude.

    Parameters:
    -----------
    data : pd.DataFrame
        The data to be used for the scatter layer.
    latitude : str
        The name of the column in `data` containing the latitude for each point.
    longitude : str
        The name of the column in `data` containing the longitude for each point.

    Returns:
    --------
    pydeck.Layer
        A PyDeck layer representing the scatter layer.
    """
    _data = data.copy()
    _data["__locations"] = list(
        zip(
            _data[longitude].tolist(),
            _data[latitude].tolist(),
        )
    )

    fill_color_series = map_series_to_colormap(
        _data[fill_color], pd.Series([255] * len(_data))
    )
    _data["__fill_color"] = fill_color_series.tolist()
    return pdk.Layer(
        "ScatterplotLayer",
        _data,
        get_position="__locations",
        get_radius=radius_scalar,
        opacity=0.5,
        get_fill_color="__fill_color",
        # get_fill_color=[255, 140, 0],
        get_line_color=[0, 0, 0],
        pickable=True,
        auto_highlight=True,
        stroked=True,
        filled=True,
    )


import shapely


def geojson(
    data: geopandas.GeoDataFrame,
    color: str,
    opacity_scalar: Optional[float] = None,
):
    if data.geometry.isnull().sum():
        raise ValueError("geometry column cannot contain null values")

    if opacity_scalar is None:
        opacity_scalar = 0.8

    opacity_value = int(opacity_scalar * 255)

    data[["color_r", "color_g", "color_b", "opacity"]] = map_series_to_colormap(
        data[color].astype(float), pd.Series([opacity_value] * len(data))
    )
    return pdk.Layer(
        "GeoJsonLayer",
        data,
        get_fill_color="[color_r, color_g, color_b, opacity]",
        get_line_color=[255, 0, 0, 125],
        get_line_width=300,
        pickable=True,
    )
