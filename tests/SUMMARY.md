# Playwright E2E Tests - Quick Start

## What Was Created

✅ **Playwright E2E test framework** for testing theme dragging and subtitle formatting

### Files Created:
```
playwright.config.js          # Playwright configuration
tests/
├── theme-drag.spec.js      # Main test file (7 tests)
├── test-helpers.js         # Utility functions
├── README.md               # Detailed documentation
└── SUMMARY.md              # This file
package.json                # Updated with test scripts
```

## Run Tests

### Quick Start (Headless)
```bash
npm test
```

### Interactive UI Mode (Recommended)
```bash
npm run test:ui
```

### Run Specific Test
```bash
npx playwright test -g "should apply color"
```

### Debug Mode
```bash
npm run test:debug
```

## Test Coverage

| Test | Description |
|------|-------------|
| Apply color formatting | Verifies JSON is created with sequence keys |
| Preserve sequence keys | **Critical**: Tests that dragging doesn't regenerate keys |
| Update timestamps | Verifies timestamps shift during operations |
| Preserve formatting | Ensures colors/styles survive timeline changes |
| JSON structure | Validates new format (sequence, timestamp, html) |
| Migration test | Confirms old format → new format conversion |
| Page reload | Verifies formatting persists across refresh |

## What Tests Verify

1. **Sequence Key Stability** ✅
   - Keys like `"160"`, `"161"` never change
   - Even after dragging timeline left/right

2. **Timestamp Updates** ✅
   - `timestamp` field updates correctly
   - Relative to theme start

3. **Formatting Preservation** ✅
   - Colors, bold, italic survive operations
   - HTML content maintained

## Next Steps

1. **Run the tests**: `npm run test:ui`
2. **Watch them execute** in real-time
3. **Debug failures** using the UI inspector
4. **Add more tests** as needed

## Notes

- Tests automatically start the server on port 5000
- Tests clean up (delete) JSON files before running
- First run downloads Chromium browser (~110MB)
- Tests take ~10-30 seconds to run
