// svelte.config.js
// import adapter from "@sveltejs/adapter-node";
import adapter from '@sveltejs/adapter-auto';
// import adapter from '@sveltejs/adapter-vercel';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	compilerOptions: {
		// Force runes mode for the project, except for libraries. Can be removed in svelte 6.
		runes: ({ filename }) => (filename.split(/[/\\]/).includes('node_modules') ? undefined : true),
		experimental: {
			async: true // REQUIRED for `await` in components / remote functions (the game)
		}
	},

	// Consult https://svelte.dev/docs/kit/integrations for more about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		adapter: adapter(),
		// Shortcutting:
		alias: {
			'$routes': './src/routes',
			'$lib': './src/lib',
			'$src': './src'
		},
		experimental: {
			remoteFunctions: true // REQUIRED by the game's *.remote.js files
		}
	},

	vitePlugin: {
		exclude: [],
		// experimental options
		inspector: {
			toggleKeyCombo: 'meta-shift',
			holdMode: false,
			showToggleButton: 'never', //always
			toggleButtonPos: 'bottom-left'
		}
	}
};

export default config;
