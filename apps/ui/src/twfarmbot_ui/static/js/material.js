// Register all Material Web components eagerly to avoid duplicate registrations
// when using esm.run CDN. CDN bundlers cannot deduplicate shared dependencies
// (like md-elevation, md-focus-ring, md-ripple) across individual component imports.
// See: https://github.com/material-components/material-web/issues/5107
import "@material/web/all.js";
