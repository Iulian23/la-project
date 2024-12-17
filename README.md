# Proiect de Predicție a Calității Aerului

Prezentare Generală

Acest proiect analizează și prezice nivelurile calității aerului folosind modele de învățare automată precum Regresia Liniară și Random Forest. Setul de date include măsurători ale calității aerului, cum ar fi arithmetic_mean și alte caracteristici precum componentele datei (day_local, month_local). Modelele sunt antrenate pentru a prezice calitatea aerului și pentru a evalua corelațiile dintre variabile.

Funcționalități Cheie

Pregătirea Datelor:

Gestionarea valorilor lipsă.

Transformarea caracteristicilor brute folosind VectorAssembler.

Modele de Învățare Automată:

Regresia Liniară: Prezice calitatea aerului pe baza caracteristicilor de intrare.

Random Forest Regressor: Îmbunătățește acuratețea prin metode de ansamblu.

Metrici de Evaluare:

Eroarea Pătratică Medie (RMSE).

Predicțiile modelelor sunt reprezentate grafic pentru vizualizare.

Analiza Corelațiilor:

Generează o matrice de corelație pentru a studia relațiile dintre variabile.

Rezultatele sunt afișate sub formă de heatmap pentru o interpretare ușoară.

Cerințe

Python 3.8+

PySpark

Pandas

Matplotlib

Seaborn

Cum se Rulează

Instalează Dependențele:

pip install pyspark pandas matplotlib seaborn

Rularea Proiectului:

Încarcă setul de date.

Rulează scriptul pentru a antrena modelele, a evalua rezultatele și a afișa vizualizările.

Rezultate Așteptate:

Metrici de performanță ale modelelor (RMSE).

Predicții alături de valorile reale.

Heatmap de corelație.

Rezultate

Regresia Liniară: Oferă predicții de bază.

Random Forest: Oferă o performanță îmbunătățită.

Heatmap-ul corelațiilor dezvăluie relațiile dintre variabile precum day_local, month_local și nivelurile calității aerului.
