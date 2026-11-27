# Mobile application source

The mobile client is a React Native application built with Expo, TypeScript, and Expo Router.

The complete cleaned Expo project is currently maintained as a separate source package and is not yet committed into this repository. This directory is reserved for its source integration.

When adding the Expo project here, preserve its app routes, components, hooks, assets, `package.json`, and lockfile. Do not commit `node_modules/`, `.expo/`, `.git/`, or local `.env` files.

For local development, set `EXPO_PUBLIC_API_URL` to the FastAPI server address reachable from the device. The current development setup runs the API on the developer's computer; it is not a public hosted endpoint.
