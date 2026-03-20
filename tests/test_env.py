import subprocess
import pytest

# Test pour vérifier l'environnement de développement
def test_uv_is_installed():
    """
    Vérifie que l'outil 'uv' est bien installé et accessible dans le PATH.
    Cela évite des échecs silencieux lors de l'exécution des outils de build ou de couverture.
    C'est aussi notre premier test, ce qui permet de ne pas avoir d'erreurs avec un dossier absent ou pas de test à exécuter
    """
    try:
        # Exécution de la commande uv --version pour tester la présence de l'exécutable
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=True)
        
        # Le code de retour doit être 0 (succès)
        assert result.returncode == 0, f"Erreur lors de l'exécution de uv: {result.stderr}"
        # On vérifie que la sortie contient bien 'uv'
        assert "uv" in result.stdout.lower(), f"La commande uv n'a pas renvoyé la version attendue: {result.stdout}"
        
    except FileNotFoundError:
        # Si la commande n'est pas trouvée du tout
        pytest.fail("L'outil 'uv' n'a pas été trouvé. Assurez-vous qu'il est installé et présent dans le PATH.")
