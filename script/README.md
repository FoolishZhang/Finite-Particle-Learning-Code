# Scientific source entrypoints

Start with ../REPRODUCING.md and ../docs/CORE_RESULT_MAP.json. Source files retain original names and mathematical implementations. The supported clean interface is ../tools/production_replay.py; legacy run/freeze/report wrappers can still depend on historical management artifacts and are not silently bypassed.

../release/SOURCE_STATUS.json distinguishes source-readable archival files from supported bounded computations. Do not run every file named test or check as a batch: archival tests can require unavailable environments, data or old management wrappers. All supported outputs go to new directories outside this source tree. Keep failed and unresolved results and do not use -O/-OO.
