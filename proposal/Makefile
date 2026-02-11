# Makefile for PhD Research Proposal
# Multi-Agent Reinforcement Learning for Cross-Chain Liquidity and Routing Optimization

.PHONY: all clean pdf view

# Default target
all: main.pdf

# Main PDF compilation
main.pdf: main.tex references.bib
	@echo "Compiling LaTeX document..."
	@tectonic main.tex 2>&1 | grep -v "note: downloading" | grep -v "note: Running" | grep -v "note: Rerunning" || true
	@echo "First pass complete."
	@tectonic main.tex 2>&1 | grep -v "note: downloading" | grep -v "note: Running" | grep -v "note: Rerunning" || true
	@echo "Second pass complete."
	@echo "PDF generated: main.pdf"
	@ls -lh main.pdf

# Quick compilation (single pass)
quick:
	@tectonic main.tex 2>&1 | grep -v "note: downloading" || true

# Clean auxiliary files
clean:
	@rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fls *.fdb_latexmk
	@rm -f *.synctex.gz *.run.xml *.bcf
	@echo "Cleaned auxiliary files."

# View PDF (requires evince or similar)
view: main.pdf
	@if command -v evince >/dev/null 2>&1; then \
		evince main.pdf & \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open main.pdf & \
	elif command -v open >/dev/null 2>&1; then \
		open main.pdf & \
	else \
		echo "No PDF viewer found. Please open main.pdf manually."; \
	fi

# Check dependencies
check:
	@echo "Checking dependencies..."
	@which tectonic >/dev/null 2>&1 && echo "[OK] tectonic found" || echo "[MISSING] tectonic not found"
	@echo "Dependencies check complete."

# Help
help:
	@echo "Available targets:"
	@echo "  make all     - Build main.pdf (default)"
	@echo "  make quick   - Single-pass compilation"
	@echo "  make clean   - Remove auxiliary files"
	@echo "  make view    - Open PDF viewer"
	@echo "  make check   - Check dependencies"
	@echo "  make help    - Show this help"
