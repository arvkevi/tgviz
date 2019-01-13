import umap

import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go

from cyvcf2 import VCF
from dash.dependencies import Input, Output
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE


with open('tgviz_description.md', 'r') as file:
    tgviz_md = file.read()


def merge(a, b):
    return dict(a, **b)


def omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


def Card(children, **kwargs):
    return html.Section(
        children,
        style=merge({
            'padding': 20,
            'margin': 5,
            'borderRadius': 5,
            'border': 'thin lightgrey solid',

            # Remove possibility to select the text for better UX
            'user-select': 'none',
            '-moz-user-select': 'none',
            '-webkit-user-select': 'none',
            '-ms-user-select': 'none'
        }, kwargs.get('style', {})),
        **omit(['style'], kwargs)
    )


def NamedInlineRadioItems(name, short, options, val, **kwargs):
    return html.Div(
        id=f'div-{short}',
        style=merge({
            'display': 'inline-block'
        }, kwargs.get('style', {})),
        children=[
            f'{name}:',
            dcc.RadioItems(
                id=f'radio-{short}',
                options=options,
                value=val,
                labelStyle={
                    'display': 'inline-block',
                    'margin-right': '7px',
                    'font-weight': 300
                },
                style={
                    'display': 'inline-block',
                    'margin-left': '7px'
                }
            )
        ],
        **omit(['style'], kwargs)
    )


tgviz_layout = html.Div(
    className="container",
    style={
        'width': '90%',
        'max-width': 'none',
        'font-size': '1.5rem',
        'padding': '10px 30px'
    },
    children=[
        # Header
        html.Div(className="row", children=[
            html.H3(
                'Population Visualizations from the 1000 Genomes Project',
                id='title',
                style={
                    'float': 'left',
                    'margin-top': '50px',
                    'margin-bottom': '0',
                    'margin-left': '7px'
                }
            ),

            html.Img(
                src="https://c2.staticflickr.com/8/7279/27816810735_6eed002807_m.jpg",
                style={
                    'height': '160px',
                    'width': '240px',
                    'float': 'right',
                    'margin-right': '100px'
                }
            )
        ]),

        # Body
        html.Div(className="row", children=[
            html.Div(className="eight columns", children=[
                dcc.Graph(
                    id='graph-3d-plot',
                    style={'height': '98vh'}
                )
            ]),

            html.Div(className="four columns", children=[
                Card([
                    NamedInlineRadioItems(
                        name="Set of AISNPs to use",
                        short="dataset",
                        options=[
                            {'label': 'Kidd 55 AISNPs', 'value': 'Kidd_55'},
                            {'label': 'Seldin 128 AISNPs', 'value': 'Seldin_128'},
                        ],
                        val='Kidd_55'),

                    NamedInlineRadioItems(
                        name="Dimensionality Reduction Algorithm",
                        short="dim-red",
                        options=[
                            {'label': ' PCA', 'value': 'pca'},
                            {'label': ' T-SNE', 'value': 'tsne'},
                            {'label': ' UMAP', 'value': 'umap'}
                        ],
                        val='pca'),

                    NamedInlineRadioItems(
                        name="Population Resolution",
                        short="pop-res",
                        options=[
                            {'label': ' Super Population', 'value': 'super_pop'},
                            {'label': ' Population', 'value': 'pop'}
                        ],
                        val='super_pop'),
                ]),
                # hold the jsonified, dim reduced DataFrame in the browser
                html.Div(id='browser-df', style={'display': 'none'}),

                Card(style={'padding': '5px'}, children=[
                    html.Div(id='div-plot-click-message',
                             style={'text-align': 'center',
                                    'margin-bottom': '7px',
                                    'font-weight': 'bold'}
                             ),

                    html.Div(id='div-plot-click-info'),
                ])
            ])
        ]),

        # tgviz Description
        html.Div(
            className='row',
            children=html.Div(
                style={
                    'width': '75%',
                    'margin': '30px auto',
                },
                children=dcc.Markdown(tgviz_md)
            )
        )
    ]
)


def tgviz_callbacks(app):
    def generate_figure_image(groups, layout):
        data = []

        for idx, val in groups:
            scatter = go.Scatter3d(
                name=idx,
                x=val.loc[:, 'x'],
                y=val.loc[:, 'y'],
                z=val.loc[:, 'z'],
                text=[idx for _ in range(val.loc[:, 'x'].shape[0])],
                textposition='top right',
                mode='markers',
                marker=dict(
                    size=4,
                    symbol='circle'
                )
            )
            data.append(scatter)

        figure = go.Figure(
            data=data,
            layout=layout
        )

        return figure

    def vcf2df(vcf_fname):
        """Convert a subsetted vcf file to pandas DataFrame
        and return sample-level population data"""
        samples = 'ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel'
        dfsamples = pd.read_csv(samples, sep='\t')
        dfsamples.set_index('sample', inplace=True)
        dfsamples.drop(columns=['Unnamed: 4', 'Unnamed: 5'], inplace=True)

        vcf_file = VCF(vcf_fname)
        df = pd.DataFrame(index=vcf_file.samples)
        for variant in vcf_file():
            df[variant.ID] = variant.gt_types

        df = df.join(dfsamples, how='outer')
        df = df.drop(columns=['pop', 'super_pop', 'gender'])

        return df, dfsamples

    def reduce_dim(df, algorithm='pca'):
        """Reduce the dimensionality of the 55 AISNPs
        :param X: One-hot encoded 1kG 55 AISNPs.
        :type X: pandas DataFrame
        :param algorithm: The type of dimensionality reduction to perform.
            One of {pca, umap, tsne}
        :type algorithm: str

        :returns: The transformed X[m, n] array, reduced to X[m, n_components] by algorithm.
        """
        ncols = len(df.columns)
        ohe = OneHotEncoder(categories=[range(4)] * ncols, sparse=False)

        n_components = 3

        X = ohe.fit_transform(df.values)
        if algorithm == 'pca':
            X_red = PCA(n_components=n_components).fit_transform(X)
        elif algorithm == 'tsne':
            # TSNE, Barnes-Hut have dim <= 3
            if n_components > 3:
                print('The Barnes-Hut method requires the dimensionaility to be <= 3')
                return None
            else:
                X_red = TSNE(n_components=n_components,
                             n_jobs=4).fit_transform(X)
        elif algorithm == 'umap':
            X_red = umap.UMAP(n_components=n_components).fit_transform(X)
        else:
            return None
        return pd.DataFrame(X_red, columns=['x', 'y', 'z'], index=df.index)

    @app.server.before_first_request
    def load_vcf_data():
        global data_dict

        # convert to pandas DataFrame
        Kidd_data, dfsamples = vcf2df("data/Kidd.55AISNP.1kG.vcf")
        Seldin_data, dfsamples = vcf2df("data/Seldin.128AISNP.1kG.vcf")

        data_dict = {
            'kidd_data': Kidd_data,
            'seldin_data': Seldin_data,
            'sample_data': dfsamples
        }

    @app.callback(Output('browser-df', 'children'),
                  [Input('radio-dataset', 'value'),
                   Input('radio-dim-red', 'value')])
    def encoded_df(dataset, algorithm):
        # Plot layout

        if dataset == 'Kidd_55':
            df = data_dict['kidd_data']
        elif dataset == 'Seldin_128':
            df = data_dict['seldin_data']

        df = reduce_dim(df, algorithm=algorithm)
        dfsamples = data_dict['sample_data']
        df = df.join(dfsamples)

        # the browser DataFrame must be stored as json
        return df.to_json(date_format='iso', orient='split')

    @app.callback(Output('graph-3d-plot', 'figure'),
                  [Input('browser-df', 'children'),
                   Input('radio-pop-res', 'value')])
    def display_3d_scatter_plot(df, pop_resolution):
        # Plot layout
        layout = go.Layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(
                    title='Component 1',
                    showgrid=True,
                    zeroline=False,
                    showticklabels=True
                ),
                yaxis=dict(
                    title='Component 2',
                    showgrid=True,
                    zeroline=False,
                    showticklabels=True
                ),
                zaxis=dict(
                    title='Component 3',
                    showgrid=True,
                    zeroline=False,
                    showticklabels=True
                )
            )
        )

        df = pd.read_json(df, orient='split')

        groups = df.groupby(pop_resolution)
        figure = generate_figure_image(groups, layout)

        return figure

    def generate_table(dataframe, max_rows=1):
        return html.Table(
            # Header
            [html.Tr([html.Th(col) for col in dataframe.columns])] +

            # Body
            [html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))]
        )

    @app.callback(Output('div-plot-click-message', 'children'),
                  [Input('graph-3d-plot', 'clickData'), ])
    def display_click_message(clickData):
        """
        Displays message shown when a point in the graph is clicked.
        :param clickData:
        """
        if clickData:
            return "Sample Selected"
        else:
            return "Click a data point to show more information about the sample."

    @app.callback(Output('div-plot-click-info', 'children'),
                  [Input('browser-df', 'children'),
                   Input('graph-3d-plot', 'clickData'), ])
    def display_click_info(df, clickData):
        df = pd.read_json(df, orient='split')
        try:
            # Convert the point clicked into float64 numpy array
            click_point_np = np.array([clickData['points'][0][i]
                                       for i in ['x', 'y', 'z']]).astype(np.float64)
            # Create a boolean mask of the point clicked, truth value exists at only one row
            bool_mask_click = df[['x', 'y', 'z']].eq(click_point_np).all(axis=1)
            # Retrieve the index of the point clicked, given it is present in the set
            if bool_mask_click.any():
                clicked_idx = df[bool_mask_click].index[0]

            return html.Div(children=[
                html.H4(children=clicked_idx),
                generate_table(pd.DataFrame(data_dict['sample_data'].loc[clicked_idx]).T)],
            )
        except (TypeError, UnboundLocalError):
            return None
