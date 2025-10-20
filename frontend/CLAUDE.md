# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the frontend component of a **Multimodal RAG (Retrieval-Augmented Generation) System** designed for industrial and engineering document retrieval. The application enables semantic search across technical documents including CAD drawings, architecture diagrams, PDFs, and engineering specifications using Vision Language Models (VLMs).

**Key Features:**
- Natural language queries for technical document retrieval
- Multimodal search supporting text + image/CAD file input
- VLM-powered semantic understanding of engineering diagrams
- Citation tracking and provenance visualization
- Structured data extraction from technical drawings
- Follow-up questions on retrieved documents

## Tech Stack

- **Framework:** React 18.3.1 with TypeScript (TSX)
- **Build Tool:** Vite 6.3.6
- **UI Components:** Custom component library in `ui/` directory (shadcn/ui-style components)
  - Radix UI primitives (@radix-ui/react-*)
  - class-variance-authority for variant management
- **Styling:** Tailwind CSS 3.4.17 with custom design tokens
- **Icons:** Lucide React
- **State Management:** React hooks (useState, custom hooks)
- **HTTP Client:** Axios 1.7.9
- **Form Handling:** react-hook-form
- **Theme Management:** next-themes

## Architecture

### Component Structure

The application follows a flat component architecture with a single main `App.tsx` and specialized feature components:

```
frontend/
├── main.tsx                   # Application entry point
├── App.tsx                    # Main application with search orchestration
├── index.html                 # HTML entry point
├── components/
│   ├── Header.tsx            # Top navigation bar
│   ├── SearchBar.tsx         # Search input with model/strategy selectors
│   ├── ResultCard.tsx        # Individual search result display
│   ├── PreviewPanel.tsx      # Right sidebar for document preview & Q&A
│   ├── EmptyState.tsx        # Empty state displays
│   └── FilterSidebar.tsx     # (Unused in current implementation)
├── ui/                       # Reusable UI primitives (40+ components)
├── hooks/                    # Custom React hooks
│   ├── useSearch.ts          # Search functionality hook
│   ├── useUpload.ts          # File upload hook
│   └── useFollowUpQuestion.ts # Follow-up Q&A hook
├── services/                 # API service layer
│   ├── api.ts                # Real API service
│   ├── mockApi.ts            # Mock API for development
│   └── index.ts              # Service exports
├── types/                    # TypeScript type definitions
│   └── index.ts              # Core data types
├── lib/                      # Utility functions
├── style/                    # Global styles
│   └── globals.css           # Tailwind + custom CSS
├── figma/                    # Figma integration utilities
└── guidelines/               # Design system guidelines (template only)
```

### Data Flow

1. **Search Orchestration (App.tsx:28-36)**: User enters query → `handleSearch()` triggers search → `useSearch` hook calls service layer → Updates `hasSearched` state → Displays results
2. **Result Selection**: Clicking a `ResultCard` updates `selectedResultId` → Renders `PreviewPanel` with selected document
3. **API Service Layer (services/)**:
   - Environment variable `VITE_USE_MOCK_API` controls mock vs real API
   - Real API proxies requests to backend via `/api` prefix (configured in vite.config.ts)
   - Mock API returns hardcoded data for development

### Key Interfaces (types/index.ts)

**SearchResult**: Core data structure for search results
- Metadata: `fileName`, `filePath`, `fileType`, `similarity`, `version`, `date`
- Citation tracking: `citationNumber`
- Visual preview: `thumbnailType` ("cad" | "pdf" | "image"), `thumbnailUrl`, `previewUrl`
- Structured data extraction: `structuredData: Array<{label, value}>`

**SearchRequest**: Search API request payload
- `query`: Search query string
- `model`: VLMModel ("gpt-4o" | "qwen-vl" | "intern-vl")
- `strategy`: RetrievalStrategy ("vector" | "hybrid" | "two-stage")
- `topK`: Number of results to return
- `filters`: Optional search filters

**SearchResponse**: Search API response
- `results`: Array of SearchResult
- `totalCount`: Total number of results
- `queryTime`: Query execution time in ms
- `model`, `strategy`: Echo back request parameters

## Design System

### Visual Language
- **Color Scheme**: Dark theme with cyan/purple gradients
  - Primary: `#00d4ff` (cyan)
  - Background: `#0a0e1a` (dark blue-black)
  - Gradient accents: primary → purple-500
- **Glass Morphism**: Extensive use of `backdrop-blur-xl` with semi-transparent backgrounds (`bg-card/60`)
- **Typography**: Conservative font sizes (11px-28px range), Chinese language support
- **Borders**: Subtle borders with low opacity (`border-border/30`)
- **Background Effects**: Radial gradients and grid patterns for depth

### Component Patterns
- All interactive cards use hover states with gradient overlays
- Selected states indicated by primary color borders + gradient backgrounds
- Badge components extensively used for metadata display
- Consistent 13px text for body, 11px for metadata, 15px for headings

## Important Technical Notes

### Current State
- **Service Layer Architecture**: Clean separation between API and UI via hooks
- **Mock/Real API Toggle**: Controlled by `VITE_USE_MOCK_API` environment variable
- **API Proxy**: Vite dev server proxies `/api/*` requests to `http://localhost:8000`
- **Type Safety**: Full TypeScript coverage with shared type definitions

### Backend Context
The parent project has a Python-based backend with:
- FastAPI document retrieval service
- Knowledge base API
- Text segmentation module
- Vectorization service
- Milvus vector database server
- Image analysis using VLMs

### Backend API Endpoints (services/api.ts)

**POST /api/search** - Search documents
- Request: `{ query, model, strategy, topK, filters }`
- Response: `{ results[], totalCount, queryTime, model, strategy }`

**POST /api/upload** - Upload document
- Content-Type: multipart/form-data
- Fields: `file`, `metadata` (JSON string)
- Response: `{ success, fileId, fileName, message }`

**POST /api/question** - Follow-up question on document
- Request: `{ documentId, question, model }`
- Response: `{ answer, citations[], confidence }`

**GET /api/preview/{documentId}** - Get document preview
**GET /api/thumbnail/{documentId}** - Get document thumbnail
**GET /api/download/{documentId}** - Download original document
**GET /api/health** - Health check endpoint

### Integration Status
- ✅ API service layer implemented with proper error handling
- ✅ Custom hooks for search, upload, and follow-up questions
- ✅ TypeScript interfaces matching backend API contracts
- ⏳ Backend endpoints need to be implemented and tested
- ⏳ Preview rendering needs actual image/CAD display logic
- ⏳ File upload UI needs to be connected to upload hook

## Component Import Patterns

**IMPORTANT**: Always use clean imports without version suffixes:

✅ **Correct:**
```tsx
import { Button } from "./ui/button";
import { Slot } from "@radix-ui/react-slot";
import { cva } from "class-variance-authority";
```

❌ **Incorrect (will cause build errors):**
```tsx
import { Slot } from "@radix-ui/react-slot@1.1.2";  // NO version numbers!
import { cva } from "class-variance-authority@0.7.1";  // NO version numbers!
```

**Standard Import Patterns:**
```tsx
// UI primitives
import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Select, SelectContent, SelectItem } from "./ui/select";

// Feature components
import { Header } from "./components/Header";
import { SearchBar } from "./components/SearchBar";

// Hooks
import { useSearch } from "./hooks";

// Types
import type { VLMModel, SearchResult } from "./types";

// Services
import { service, apiService, mockApiService } from "./services";
```

## Styling Conventions

### Tailwind Configuration
- **Content Paths**: Configured to scan `./components/**`, `./ui/**`, `./hooks/**`, `./lib/**`
  - **CRITICAL**: Patterns must NOT match `node_modules` (performance issue)
- **Color Variables**: Use CSS variables directly via `var(--primary)`, not `hsl()` wrapper
  - Example: `border: "var(--border)"` not `border: "hsl(var(--border))"`
- **Custom Colors**: Defined in `style/globals.css` with proper CSS variable format

### Component Styling
- Use Tailwind utility classes exclusively (no CSS modules)
- Consistent spacing: `p-5` for card padding, `gap-3`/`gap-4` for flex gaps
- Rounded corners: `rounded-xl` (12px) or `rounded-2xl` (16px) for cards
- Text sizing follows strict hierarchy: 11px → 12px → 13px → 15px → 20px → 28px
- Always include `text-muted-foreground` for secondary text
- Gradient buttons: `bg-gradient-to-r from-primary to-purple-500`

### CSS Variable Usage
```css
/* globals.css must include Tailwind directives */
@tailwind base;
@tailwind components;
@tailwind utilities;

/* Color variables should be direct values (not HSL strings) */
:root {
  --primary: #00d4ff;              /* ✅ Works with Tailwind */
  --background: #0a0e1a;
  --border: rgba(0, 212, 255, 0.15);
}

/* Tailwind config references these directly */
colors: {
  primary: "var(--primary)",        /* ✅ Correct */
  // NOT: "hsl(var(--primary))"     /* ❌ Wrong format */
}
```

## Extending the Application

When adding new features:

1. **New Components**: Place in `components/` directory, follow existing naming conventions
2. **UI Primitives**: Use existing components from `ui/` directory (comprehensive set available)
3. **State Management**:
   - Use custom hooks for API interactions (see `hooks/`)
   - Keep UI state in `App.tsx` for shared data, local component state for UI-only state
4. **API Integration**:
   - Add new endpoints to `services/api.ts`
   - Create corresponding mock implementations in `services/mockApi.ts`
   - Define TypeScript interfaces in `types/index.ts`
   - Wrap API calls in custom hooks for reusability
5. **Styling**:
   - Match the existing glass morphism + gradient aesthetic
   - Use defined CSS variables for colors
   - Never use inline `@apply` with classes that don't exist (causes build errors)
6. **Typography**: Maintain Chinese language support and existing font size hierarchy

## Common Issues & Solutions

### Build Errors

**Error: "The `outline-ring/50` class does not exist"**
- **Cause**: Using `@apply` with invalid Tailwind classes or opacity syntax
- **Solution**: Use standard Tailwind classes or define custom CSS properties

**Error: "Failed to resolve import @radix-ui/react-slot@1.1.2"**
- **Cause**: Version numbers in import statements
- **Solution**: Remove version suffixes from all imports

**Warning: "content configuration includes pattern matching node_modules"**
- **Cause**: Overly broad glob pattern like `./**/*.ts`
- **Solution**: Use specific paths like `./components/**/*.tsx`, `./ui/**/*.tsx`

**Error: "hsl(var(--border)) not working"**
- **Cause**: CSS variables defined as hex/rgba, but Tailwind config expects HSL
- **Solution**: Use `var(--border)` directly in Tailwind config, not `hsl(var(--border))`

## File Locations Reference

- Main application entry: `main.tsx`
- Main application logic: `App.tsx`
- Custom hooks: `hooks/useSearch.ts`, `hooks/useUpload.ts`, `hooks/useFollowUpQuestion.ts`
- API services: `services/api.ts` (real), `services/mockApi.ts` (mock)
- Type definitions: `types/index.ts`
- Tailwind config: `tailwind.config.js`
- Global styles: `style/globals.css`
- Vite config: `vite.config.ts` (includes API proxy settings)
