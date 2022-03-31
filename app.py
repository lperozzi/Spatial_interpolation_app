import dash

import pandas as pd

from dash import Input, Output, dcc, html

import os

import dash_vtk
import dash_bootstrap_components as dbc
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import pandas as pd
import random

import numpy as np
import plotly.express as px
import pyvista as pv

from vtk.util.numpy_support import vtk_to_numpy

# from dash_vtk.utils import presets
# print(presets)
random.seed(42)


path = "data/"
bor = pv.read("data/sondage.vtk")
df = pd.read_csv('data/sondage_clean.csv')


def updateWarp_surface( name='Nearest'):
    t = os.path.join(path,name + "." + "vtk")
    mesh = pv.read(t)
    polydata = mesh.extract_geometry()
    points = polydata.points.ravel()
    polys = vtk_to_numpy(polydata.GetPolys().GetData())
    elevation = polydata["Elevation"]
    min_elevation = np.amin(elevation)
    max_elevation = np.amax(elevation)
    return [points, polys, elevation, [min_elevation, max_elevation]]

points, polys, elevation, color_range1 = updateWarp_surface()


def updateWarp_surface2( name='SGS_real0'):
    t = os.path.join(path,name + "." + "vtk")
    mesh = pv.read(t)
    polydata = mesh.extract_geometry()
    points = polydata.points.ravel()
    polys = vtk_to_numpy(polydata.GetPolys().GetData())
    elevation = polydata["Elevation"]
    min_elevation = np.amin(elevation)
    max_elevation = np.amax(elevation)
    # cmap='BrBG'
    # visibility= 1
    if name == 'SGS_local_stdev':
        cmap = 'BuRd'
        min_elevation = 0
        max_elevation = 30
    else:
        cmap = 'BrBG'
    return [points, polys, elevation, [min_elevation, max_elevation],cmap]

points, polys, elevation, color_range2, cmap = updateWarp_surface2()


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

server = app.server

vtk_view = dash_vtk.View(
    id="vtk-view",
    
    pickingModes=["hover"],
    background=[128.0/255.0, 128.0/255.0,128.0/255.0 ],
    children=[
        dash_vtk.GeometryRepresentation(
            id="vtk-representation",
            children=[
                dash_vtk.PolyData(
                    id="vtk-polydata",
                    points=points,
                    polys=polys,
                    children=[
                        dash_vtk.PointData(
                            [
                                dash_vtk.DataArray(
                                    id="vtk-array",
                                    registration="setScalars",
                                    name="elevation",
                                    values=elevation,
                                )
                            ]
                        )
                    ],
                )
            ],
            colorMapPreset="BrBG",
            colorDataRange=color_range1,
            property={"edgeVisibility": False},
            showCubeAxes=True,
            cubeAxesStyle={"axisLabels": ["E", "N", "Depth"],
            "axisTextStyle": {
                            "fontColor": 'white',
                            "fontStyle": 'normal',
                            "fontSize": 8,
                            },
                        "tickTextStyle": {
                            "fontColor": 'white',
                            "fontStyle": 'normal',
                            "fontSize": 8,
                            },},
        ),
         
        dash_vtk.PointCloudRepresentation(
            xyz=bor.points.ravel(),
            scalars=bor['Elevation'],
            colorDataRange=color_range1,
            colorMapPreset='BrBG',
            property={"pointSize": 10, "edgeVisibility": True,"lineWidth":10},
        ),
    ],
    cameraPosition=[0,0,1],
    cameraViewUp=[0, 1, 0],
    cameraParallelProjection=False,
)

vtk_view2 = dash_vtk.View(
    id="vtk-view2",
       
    background=[128.0/255.0, 128.0/255.0,128.0/255.0 ],
    pickingModes=["hover"],
    children=[
        dash_vtk.GeometryRepresentation(
            id="vtk-representation2",
            children=[
                dash_vtk.PolyData(
                    id="vtk-polydata2",
                    points=points,
                    polys=polys,
                    children=[
                        dash_vtk.PointData(
                            [
                                dash_vtk.DataArray(
                                    id="vtk-array2",
                                    registration="setScalars",
                                    name="elevation",
                                    values=elevation,
                                )
                            ]
                        )
                    ],
                )
            ],
            colorMapPreset=cmap,
            colorDataRange=color_range2,
            property={"edgeVisibility": False},
            showCubeAxes=True,
            cubeAxesStyle={"axisLabels": ["E", "N", "Depth"],
            "axisTextStyle": {
                            "fontColor": 'white',
                            "fontStyle": 'normal',
                            "fontSize": 8,
                            },
                        "tickTextStyle": {
                            "fontColor": 'white',
                            "fontStyle": 'normal',
                            "fontSize": 8,
                            },},
        ),
        dash_vtk.PointCloudRepresentation(
            xyz=bor.points.ravel(),
            scalars=bor['Elevation'],
            colorDataRange=color_range2,
            colorMapPreset='BrBG',
            property={"pointSize": 10, "edgeVisibility": True,"lineWidth":10,
                       "opacity":1},
        ),

 
    ],
    cameraPosition=[0,0,1],
    cameraViewUp=[0, 1, 0],
    cameraParallelProjection=False,
)

dff = df[["lat","lon",'MOLASSE']]
dff['size'] = dff['MOLASSE'] *-1
fig = px.scatter_mapbox(
        dff,
        lat="lat",
        lon="lon",
        color='MOLASSE',
        color_continuous_scale='earth',
        size='size',
        size_max=10,
        zoom=10
        
    )
fig.update_layout(margin=dict(l=20, r=20, t=10, b=20),
mapbox = dict(
                            uirevision='no reset of zoom', # prevent to dezoom 
                            # accesstoken = token,
                            style = "https://api.maptiler.com/maps/ch-swisstopo-lbm-grey/style.json?key=Y0QFvc1p5eTvlCMj4GEX",
                            zoom= 11,
                            center= {"lon": 6.115, "lat": 46.215},
                        ),)

figura = dbc.Card(
    [
        html.Div(
            [
                dcc.Graph(figure=fig,style={"height": "30vh", "width": "100%"}),                    
            ]
        ),
        

    ],
    body=True,
)


text = dbc.Card(
    dbc.CardBody(
        [
            html.P(
                "Spatial interpolation is the process of using points with known values to estimate values at other unknown points. This app present different approaches to interpolate the depth of Molasse measured in 63 wells over the Geneva area in Switzerland. The deterministic interpolation section allow to explore 3 different algorithms (nearest neighbors,  biharmonic spline and Gaussian processes), while in the geostatistical approach, an ordinary Kriging and a Sequential Gaussian Simulation (SGS) algorithms have been used. It seems that Kriging and SGS are the better estimator and that SGS, by computing local summary statistics, allow to quantify the local uncertainty. In addition, SGS results better represents the natural variability of the Molassse surface.",
                style={'color': '#1E293B', 'fontSize': 14, "font-family":"Product Sans"},
            ),
            dcc.Markdown(
                """
                The codes to build this app and the notebooks for the spatial analysis are available [here](https://github.com/lperozzi/Spatial_interpolation_app) and [here](https://github.com/lperozzi/Spatial_interpolation_analysis), respectively.
                 """,
                style={'color': '#1E293B', 'fontSize': 14, "font-family":"Product Sans"},
            ),

            dbc.Button("#scikit-learn", 
                        color="dark", 
                        className="me-1",
                        size="sm",
                        style={"color": "#F0FDFA",
                                "background-color": "#134E4A",
                                "border-color": "#F0FDFA",
                                "border-radius":"9999px",
                                'fontSize': 14,
                                "font-family":"Product Sans",
                                "font-weight":500,
                                "padding": "0.1rem 1rem 0.1rem 1rem"},
                        href="https://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process"),
            dbc.Button("#verde", 
                        color="dark", 
                        className="me-1",
                        size="sm",
                        style={"color": "#F0FDFA",
                                "background-color": "#134E4A",
                                "border-color": "#F0FDFA",
                                "border-radius":"9999px",
                                'fontSize': 14,
                                "font-family":"Product Sans",
                                "padding": "0.1rem 1rem 0.1rem 1rem"},
                        href="https://www.fatiando.org/verde/latest/"),
            dbc.Button("#geostatpy", 
                        color="dark", 
                        className="me-1",
                        size="sm",
                        style={"color": "#F0FDFA",
                                "background-color": "#134E4A",
                                "border-color": "#F0FDFA",
                                "border-radius":"9999px",
                                'fontSize': 14,
                                "font-family":"Product Sans",
                                "padding": "0.1rem 1rem 0.1rem 1rem"},
                        href="https://github.com/GeostatsGuy/GeostatsPy"),
            dbc.Button("#dash-vtk", 
                        color="#747474", 
                        className="me-1",
                        size="sm",
                        style={"color": "#F0FDFA",
                                "background-color": "#134E4A",
                                "border-color": "#F0FDFA",
                                "border-radius":"9999px",
                                'fontSize': 14,
                                "font-family":"Product Sans",
                                "padding": "0.1rem 1rem 0.1rem 1rem"},
                        href="https://dash.plotly.com/vtk"),
            
            html.P(
                "Source of data : SITG ; Author: Lorenzo Perozzi",
                className="mt-4",
                style={'color': '#4B5563', 'fontSize': 12, 
                "font-family":"Product Sans"},
            ),

        ]
    ),
    style={"width": "100%"},
)

controls_det = dbc.Card(
    [
        html.Div(
            [
                html.H4("Deterministic interpolation",style={'color': '#444444', "font-family":"Product Sans"},),
                dcc.Dropdown(
                    id="dataset",
                    options=[
                         {"label": "Nearest-neighbor interpolation", "value": "Nearest"},
                         {"label": "Biharmonic spline interpolation (overfit)", "value": "Spline_overfit"},
                         {"label": "Biharmonic spline interpolation (smooth)", "value": "Spline"},
                         {"label": "Gaussian Processes interpolation", "value": "GaussianProcess"}
                    ],
                    style={'fontSize': 14,"font-family":"Product Sans"},
                    value="Nearest",
                    clearable=False,
                ),
            ]
        ),
        html.Div(
            [
                dcc.Checklist(
                        id="toggle-cube-axes1",
                        options=[
                            {"label": " Show axis grid", "value": "grid"},
                        ],
                        value=["grid"],
                        labelStyle={"display": "inline-block","font-family":"Product Sans", 'fontSize': 14,},
                    ),
            ]
        ),

       
    ],
    body=True,
)


controls_sgs = dbc.Card(
    [
        html.Div(
            [
                html.H4("Geostatistical interpolation",style={'color': '#444444', "font-family":"Product Sans"},),
                dcc.Dropdown(
                    id="dataset2",
                    options=[
                        {"label": "Kriging", "value": "OK"},
                        {"label": "SGS Realization 1 of 10", "value": "SGS_real0"},
                        {"label": "SGS Realization 2 of 10", "value": "SGS_real4"},
                        {"label": "SGS Realization 3 of 10", "value": "SGS_real8"},
                        {"label": "SGS average realization", "value": "SGS_etype"},
                        {"label": "SGS variance of realizations (high (red) to low (blue) variance)", "value": "SGS_local_stdev"},
                    ],
                    style={'fontSize': 14,"font-family":"Product Sans"},
                    value="SGS_etype",
                    clearable=False,
                ),
            ]
        ),

        html.Div(
            [
                dcc.Checklist(
                        id="toggle-cube-axes2",
                        options=[
                            {"label": " Show axis grid", "value": "grid"},
                        ],
                        value=["grid"],
                        labelStyle={"display": "inline-block","font-family":"Product Sans",'fontSize': 14,},
                    ),
            ]
        ),

       
    ],
    body=True,
    
)

explanation = html.B("Exploring spatial interpolation",style={'color': '#444444', "font-family":"Montserrat", "font-weight":700},)
LOGO = "https://raw.githubusercontent.com/lperozzi/personale_web/master/public/static/images/logo_main.png"

search_bar = dbc.Row(
    html.A(
    [
        dbc.Col(html.Img(src=LOGO, height="40px")),
    ],
    href="https://www.geomaap.io/",
    style={"textDecoration": "none"},
    ),
    className="g-0 ms-auto flex-nowrap mt-1 mt-md-0",
    align="center",

),

navbar = dbc.Navbar(
    dbc.Container(
        [
                dbc.Row(
                    [
                        dbc.Col(dbc.NavbarBrand(html.H1(explanation), className="ms-2")),
                    ],
                    align="center",
                    className="g-2",
                ),

            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                search_bar,
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color="light",
    dark=False,
    className="pt-0 pb-0"
)



app.layout = dbc.Container(
    children=[
        navbar,
        dbc.Row(html.Hr()),        
        dbc.Row(
            [
                dbc.Col(text, md=6),
                dbc.Col(figura, md=6),
                
            ],
            align="fluid",
            style={"height":"35%", }
        ),
        dbc.Row(
            [
                dbc.Col(controls_det,md=6),
                dbc.Col(controls_sgs,md=6),
            ],
            style={"height":"8%"}

        ),
        dbc.Row(
            [
                html.Div(vtk_view, style={"height": "80%", "width": "50%"}),
                html.Div(vtk_view2, style={"height": "80%", "width": "50%"}),
            ],
            align="center",
            style={"height": "45%"},
        ),

    ],
    fluid=False,
    style={"height": "100vh"},
)

@app.callback(
    [
        Output("vtk-representation", "showCubeAxes"),
        Output("vtk-representation", "colorDataRange"),
        Output("vtk-polydata", "points"),
        Output("vtk-polydata", "polys"),
        Output("vtk-array", "values"),
        Output("vtk-view", "triggerResetCamera"),
    ],
    [
        Input("dataset", "value"),
        Input("toggle-cube-axes1", "value"),
    ],
)
def updatePresetName(  fname, cubeAxes):
    points, polys, elevation, color_range1 = updateWarp_surface( fname)
    
    return [
        "grid" in cubeAxes,
        color_range1,
        points,
        polys,
        elevation,
        random.random(),
        
    ]

@app.callback(
    [
        Output("vtk-representation2", "showCubeAxes"),
        Output("vtk-representation2", "colorDataRange"),
        Output("vtk-representation2", "colorMapPreset"),
        Output("vtk-polydata2", "points"),
        Output("vtk-polydata2", "polys"),
        Output("vtk-array2", "values"),
        Output("vtk-view2", "triggerResetCamera"),
    ],
    [
        Input("dataset2", "value"),
        Input("toggle-cube-axes2", "value"),
    ],
)
def updatePresetName2(  fname, cubeAxes):
    points, polys, elevation, color_range2, cmap = updateWarp_surface2( fname)
    
    return [
        "grid" in cubeAxes,
        color_range2,
        cmap,
        points,
        polys,
        elevation,
        random.random(),
        
    ]


if __name__ == "__main__":
    app.run_server(debug=True)