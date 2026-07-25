# Advanced Temperature Control

Different teas need different water temperatures, and the Teapot API lets you
control this precisely per brew request.

{* ../../docs_src/brewing/tutorial001.py hl[4] *}

/// tip

Keep water below boiling for green tea. 80 degrees Celsius preserves the
delicate flavor compounds that boiling water destroys.

///

{!../partials/temperature-table.md!}

Black tea tolerates a full boil, while white tea prefers around 75 degrees.
The `temperature_c` field accepts any integer between 60 and 100.
