import React from 'react';

/**
 * Keeps one broken panel from taking down the console.
 *
 * The map is the most fragile thing on the screen: it depends on a third-party
 * tile server and on browser APIs that a locked-down government desktop may or
 * may not expose. If it fails during a live demo, the duty officer should still
 * see the warnings, the table and the bulletin.
 */
export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // eslint-disable-next-line no-console
    console.error(`[${this.props.label || 'panel'}] render failed`, error, info?.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="panel">
        <div className="panel-head">
          <span className="panel-title">{this.props.label || 'Panel'} unavailable</span>
          <button className="btn" onClick={() => this.setState({ error: null })}>retry</button>
        </div>
        <div className="panel-body small">
          {this.props.fallback
            || 'This panel failed to render. The rest of the console keeps working; every value shown here is also in the warning table below.'}
          <div className="tiny muted mono" style={{ marginTop: 6 }}>
            {String(this.state.error?.message || this.state.error).slice(0, 200)}
          </div>
        </div>
      </div>
    );
  }
}
