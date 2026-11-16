# Public sharing mirror

`xjtuyanshi/ra2-hd-assets` is the private development repository.
`xjtuyanshi/ra2-hd-public` is its public sharing mirror. Download tools use the public
repository; development credentials never belong in a sharing package.

The owner chose **on-demand synchronization**. Private commits and releases do not
automatically publish anything. When another share is requested:

1. Inspect private `main`, other remote branches/tags and all published releases. Check
   for accidental credentials and preserve unrelated local work. Never publish `dev`,
   machine-local files, unreleased local art or original game files.
2. Fast-forward corresponding public references. Do not force unrelated public work away.
   Commit necessary download/entry documentation changes to both repositories so their
   corresponding source references stay identical.
3. Copy every published release attachment, including incremental manifests' base
   releases and the optional environment backup. Preserve names, sizes and SHA-256
   digests. Publish a destination release only after its inventory matches. Descriptions
   can explain the public mirror instead of claiming it is private.
4. Recheck source references after copying, compare every asset's size/digest, and
   anonymously verify the public entry and download endpoints. Keep the private source
   private; remove temporary transfer credentials and branches.

[START_HERE.md](START_HERE.md) explicitly selects the public download repository,
overriding private repository fields in archived incremental manifests. The manifests
and chunks remain byte-identical; their historical repository field is not an access
requirement when the override is supplied.

Mirroring does not validate Windows, merge experimental branches, or imply that
unpublished branch-specific art was backed up.
