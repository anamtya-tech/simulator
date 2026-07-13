#!/bin/bash
# Auto-commit every hour if there are changes

REPO_DIR="/home/makes/Git_Dev/simulator"
BRANCH="makesh_vm"
REMOTE="origin"
README="$REPO_DIR/CHANGELOG_README.md"

cd "$REPO_DIR" || exit 1

# Ensure branch exists and is checked out
git checkout $BRANCH || git checkout -b $BRANCH

while true; do
    # Collect changes
    CHANGES=$(git status --short)

    if [ -n "$CHANGES" ]; then
        # Timestamp
        STAMP=$(date +"%Y-%m-%d_%H-%M")

        # Commit message
        MESSAGE="[$STAMP] Code updates on VM: $CHANGES"

        # Stage and commit
        git add .
        git commit -m "$MESSAGE"

        # Update README with list of changes
        {
          echo "## Commit on $STAMP"
          echo ""
          echo "$CHANGES"
          echo ""
        } >> "$README"

        # Push to remote
        git push $REMOTE $BRANCH
        echo "Committed and pushed at $STAMP"
    else
        echo "[$(date +"%Y-%m-%d_%H-%M")] No changes, skipping commit."
    fi

    # Sleep for 1 hour
    sleep 3600
done
