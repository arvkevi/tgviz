# -*- coding: utf-8 -*-
import dash

from tgviz_app import tgviz_layout, tgviz_callbacks

app = dash.Dash(__name__)
server = app.server

app.scripts.append_script({'external_url': 'https://codepen.io/plotly/pen/BGyZNa.js'
                           })

# App
app.layout = tgviz_layout


# Callbacks
tgviz_callbacks(app)


# Load external CSS
external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
    "//fonts.googleapis.com/css?family=Raleway:400,300,600",
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css"
]

for css in external_css:
    app.css.append_css({"external_url": css})

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True)
