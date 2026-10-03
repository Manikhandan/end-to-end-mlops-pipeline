# Incident response for inference errors

If `/ready` is false, a model was never promoted. Train first.

If JSON payloads fail validation, the client contract changed — do not relax schemas silently.

# Feature store note

This pipeline stores splits as npz. Treat `var/store` as ephemeral unless you snapshot it.
