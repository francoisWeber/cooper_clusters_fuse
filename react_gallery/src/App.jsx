import React, { Component } from 'react';
import { hot } from 'react-hot-loader';
// import HelloWorld from './components/hello-world';
import Gallery from 'react-photo-gallery';
import './styles/app.css';

// load the clustering.json config file
const clustering = require('../../output/clustering.json');


class App extends Component {
  constructor(props) {
    super(props);
    this.state = {};
  }

  render() {
    return (
      <div>
        {clustering.map(event => {
          return (
            <div>
              <h1>Evenement (cluster #{event.cluster})</h1>
              <Gallery photos={event.photos} columns={5} />
            </div>
          );
        })}
      </div>
    );
  }
}

export default hot(module)(App);
