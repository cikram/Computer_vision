import cv2
import numpy as np

import fonctions.TraitementImage as ti
import fonctions.dessin as ds


# ============================================================
# VARIABLES GLOBALES
# ============================================================

dessin = False

couleur_final = (0, 255, 0)      # Vert
couleur_dessin = (0, 0, 255)     # Rouge

largeur, hauteur = 320, 480

points = []

img = None
img_bk = None
warped_image = None
tiroc = None


# ============================================================
# GESTION DE LA SOURIS
# ============================================================

def on_mouse(event, x, y, flags, param):

    global points, dessin
    global img, img_bk
    global warped_image

    # --------------------------------------------------------
    # CLIC GAUCHE
    # --------------------------------------------------------

    if event == cv2.EVENT_LBUTTONDOWN:

        dessin = True

        # Ajouter le point cliqué
        add_point(points, (x, y))

        # ----------------------------------------------------
        # SI ON A 4 POINTS
        # ----------------------------------------------------

        if len(points) == 4:

            # Fermer le polygone
            dess_polygone(
                img,
                points,
                (x, y),
                True
            )

            # On arrête le dessin temporaire
            dessin = False

            # ------------------------------------------------
            # IMAGE SOURCE
            # ------------------------------------------------

            # Copie propre de l'image
            img_bk = img.copy()

            # ------------------------------------------------
            # POINTS SOURCE
            # ------------------------------------------------

            src = np.float32(points)

            # ------------------------------------------------
            # POINTS DESTINATION
            # ------------------------------------------------

            dst = np.float32([
                [0, 0],
                [largeur - 1, 0],
                [largeur - 1, hauteur - 1],
                [0, hauteur - 1]
            ])

            # ------------------------------------------------
            # CALCUL DE LA MATRICE DE TRANSFORMATION
            # ------------------------------------------------

            M = cv2.getPerspectiveTransform(
                src,
                dst
            )

            # ------------------------------------------------
            # APPLICATION DE LA TRANSFORMATION
            # ------------------------------------------------

            warped_image = cv2.warpPerspective(
                img_bk,
                M,
                (largeur, hauteur)
            )

            # ------------------------------------------------
            # AFFICHAGE DU RESULTAT
            # ------------------------------------------------

            cv2.imshow(
                "Perspective Warping",
                warped_image
            )

            # ------------------------------------------------
            # RESET DES POINTS
            # ------------------------------------------------

            points.clear()


    # --------------------------------------------------------
    # MOUVEMENT DE LA SOURIS
    # --------------------------------------------------------

    elif event == cv2.EVENT_MOUSEMOVE:

        if dessin == True:

            # Revenir à une image propre
            img = img_bk.copy()

            # Redessiner le polygone
            # et la ligne vers la souris
            dess_polygone(
                img,
                points,
                (x, y)
            )


# ============================================================
# AJOUT D'UN POINT
# ============================================================

def add_point(points, curt_pt):

    print(
        "Adding point #%d with position(%d,%d)"
        % (
            len(points),
            curt_pt[0],
            curt_pt[1]
        )
    )

    points.append(curt_pt)


# ============================================================
# DESSIN DU POLYGONE
# ============================================================

def dess_polygone(
    img,
    points,
    curt_pt,
    is_final=False
):

    if len(points) > 0:

        # ----------------------------------------------------
        # POLYGONE NON TERMINE
        # ----------------------------------------------------

        if is_final == False:

            # Dessiner les lignes entre les points
            ds.dess_polylignes(
                img,
                np.array([points]),
                False,
                couleur_final
            )

            # Dessiner la ligne entre le dernier
            # point et la souris
            ds.dess_ligne(
                img,
                points[-1],
                curt_pt,
                couleur_dessin
            )

        # ----------------------------------------------------
        # POLYGONE TERMINE
        # ----------------------------------------------------

        else:

            # Fermer le polygone
            ds.dess_polylignes(
                img,
                np.array([points]),
                True,
                couleur_final
            )

        # ----------------------------------------------------
        # DESSINER LES POINTS
        # ----------------------------------------------------

        for point in points:

            ds.dess_cercle(
                img,
                point,
                2,
                couleur_final,
                2
            )


# ============================================================
# AFFICHER LES INSTRUCTIONS
# ============================================================

def print_instruction(img):

    txtInstruction = (
        "Left click to specify four points to warp image. "
        "ESC to exit, 's' to save"
    )

    # On utilise uniquement les paramètres
    # compatibles avec ta fonction dess_text
    ds.dess_text(
        img,
        txtInstruction,
        (10, 20),
        0.5,
        (255, 255, 255)
    )

    print(txtInstruction)


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CHARGEMENT DE L'IMAGE
    # --------------------------------------------------------

    title = "Original Image"

    tiroc = ti.TraitementImage(
        title,
        'ressources/book_perspect1.jpg'
    )

    # Récupérer l'image
    img = tiroc.image

    # Copie de l'image originale
    img_bk = img.copy()

    # Pas encore de résultat
    warped_image = None

    # --------------------------------------------------------
    # AFFICHER LES INSTRUCTIONS
    # --------------------------------------------------------

    print_instruction(img)

    # --------------------------------------------------------
    # CREER LA FENETRE
    # --------------------------------------------------------

    cv2.namedWindow(title)

    # Afficher l'image
    cv2.imshow(title, img)

    # --------------------------------------------------------
    # ASSOCIER LA SOURIS A LA FONCTION on_mouse
    # --------------------------------------------------------

    cv2.setMouseCallback(
        title,
        on_mouse
    )

    # --------------------------------------------------------
    # BOUCLE PRINCIPALE
    # --------------------------------------------------------

    while True:

        # Afficher l'image
        cv2.imshow(
            title,
            img
        )

        # Attendre une touche
        ch = cv2.waitKey(10)

        # ----------------------------------------------------
        # ESC → QUITTER
        # ----------------------------------------------------

        if (ch & 0xFF) == 27:

            break

        # ----------------------------------------------------
        # S → SAUVEGARDER
        # ----------------------------------------------------

        elif ch == ord('s'):

            # Vérifier qu'un warp a été réalisé
            if warped_image is not None:

                filepath = 'ressources/book_sauv.jpg'

                cv2.imwrite(
                    filepath,
                    warped_image
                )

                print(
                    "File saved to " + filepath
                )

            else:

                print(
                    "Please select 4 points first."
                )

    # --------------------------------------------------------
    # FERMER LES FENETRES
    # --------------------------------------------------------

    cv2.destroyAllWindows()