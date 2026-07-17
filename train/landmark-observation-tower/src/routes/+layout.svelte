<script lang="ts">
// routes/+layout.svelte  
// import { dev, browser } from '$app/environment';
// Debug the $page.url: // import Debug from './Debug.svelte'; //usage: <Debug />
</script>

<slot />

<style>
/* :global(canvas) { position: fixed; }  */
/*
Quick and dirty: stackoverflow.com/questions/40723277/fullscreen-inverted-luminance-filter-for-dark-themed-browsing-but-keeping-hues-i

html {filter: invert(100%) hue-rotate(180deg);background:black}
img {filter: invert(100%) hue-rotate(180deg) !important;}
video {filter: invert(100%) hue-rotate(180deg) !important;} */

@media (prefers-color-scheme: dark) {
  :global(:root) { 
    --bg: #383838;
    --color: silver;
    /* --svg-primary: #ccc; specifically routes/(app)/Header.svelte */
    --dark: invert(100%) hue-rotate(180deg);
  }

  :global(a), :global(button), :global(input), :global(textarea), :global(select), :global(option), :global(progress), :global(meter){
    filter: none;
  }
}

:global(body) {
  background-color: var(--bg, inherit);
  color: var(--color, #383838);
  filter: none!important;
}
  
:global(label) { display: block }

/* :global(:root) {  */
  /* --svg-primary: #383838; */
/* } */




:global(button) {
  outline: none; /* Remove default blue glow for consistency */
}

:global(button:focus),
:global(button:focus-visible) {
  border-color: white;
}

:global(button:not(:focus):not(:focus-visible)) {
  border-color: transparent;
}






/* accent hover-black 
.accent { color: hsl(261, 93%, 58%) } */

:global(.b--darkmode) { border-color: var(--b--darkmode, #f8f8f8 ); }

/* :global(.bg--darkmode) {
	background-color: var(--bg-darkmode, #f8f8f8 );
} */

:global(.bg-darkmode) {
  background-color: var(--bg-darkmode, rgba(188,188,188, 0.7));
}

:global(.darkmode) {
  color: var(--color);
}



:global(:root) {
	--color: inital;
}

/*
:root {color-scheme: light dark}
usage: https://dev.to/inspiredlabs/comment/1oepj */
@media (prefers-color-scheme: dark) {
	:global(:root) {
		/* --svg-primary: white; */
		--b--darkmode: #222;
		--bg-darkmode:rgba(0,0,0, 0.7);
		--bg: #383838;
		--color: white;
	}
}
</style>