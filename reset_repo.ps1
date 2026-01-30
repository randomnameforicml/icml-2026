
# 1. Initialize Git (in case it wasn't)
git init

# 2. Set Anonymous Identity for this repository ONLY
# We use fake generic info to satisfy Git's requirements without revealing your identity
git config user.email "anonymous@icml2026.conf"
git config user.name "Anonymous Submission"

# 3. Set the TARGET remote URL explicitly
git remote remove origin 2>$null
git remote add origin https://github.com/randomnameforicml/icml-2026.git

# 4. Create a fresh Orphan Branch (history-free)
git checkout --orphan temp_clean_branch_icml 2>$null

# 5. Clear the old git index (unstage everything)
# Note: if this is a fresh init, this might fail, so we ignore errors
git rm -rf --cached . 2>$null

# 6. Add files back (respecting the new .gitignore)
git add .

# 7. Create the "First" Commit
git commit -m "Initial submission for ICML 2026"

# 8. Rename current branch to main (force delete old main if exists)
git branch -D main 2>$null
git branch -m main

# 9. Force Push to your specific URL
Write-Host "Force pushing to https://github.com/randomnameforicml/icml-2026 ..."
git push -f origin main

Write-Host "----------------------------------------------------------------"
Write-Host "Success! The repository has been reset and pushed."
