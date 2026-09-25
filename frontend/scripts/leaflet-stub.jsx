/**
 * Headless stand-in for react-leaflet.
 *
 * Leaflet needs real layout and a canvas context, neither of which jsdom has, so
 * the smoke test swaps the map primitives for inert equivalents. Everything else
 * in MapPanel (layer logic, colour scales, tooltips, legend) still executes.
 */
import React from 'react';

const Box = ({ children }) => <div data-leaflet-stub>{children}</div>;

export const MapContainer = Box;
export const TileLayer = Box;
export const GeoJSON = Box;
export const CircleMarker = Box;
export const Rectangle = Box;
export const Tooltip = Box;
export const Marker = Box;
export const Popup = Box;
export const LayerGroup = Box;
export const useMap = () => ({});
