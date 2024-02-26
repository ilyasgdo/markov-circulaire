//
let selects = $( ".selecteur_loi" );
let selecteurs_unites = $( ".selecteur_unite" );
for (let i=0; i< selects.length; i++){
	console.log(i);
    let select = document.getElementById('select_loi_' + i);
    let select_unit = document.getElementById('select_unit_' + i);
	select.addEventListener("click", (event) => {
        let loi = select.options[select.selectedIndex].value;
		let unite = select_unit.options[select_unit.selectedIndex].value;
        document.getElementById("image_circulaire_" + i).src = "static/images/plot_circulaire_" + loi + "_" + i + "_" + unite + ".png";
    })
}

for (let i=0; i< selecteurs_unites.length; i++){
	
    let select = document.getElementById('select_unit_' + i);
    select.addEventListener("click", (event) => {
		
		let select_loi = document.getElementById('select_loi_' + i);
        let unite = select.options[select.selectedIndex].value;
		// On récupère les données de l'API - résumés circulaires
		$.get( "./api/resumes_circulaires/" + i + "/" + unite, function( data ) {
			// $( ".result" ).html( data );
			// alert( "Load was performed." );
			var data = data.replace(/'/g, '"');
			var donnesEtats = JSON.parse(data)
			var index = 0 ;
			for (var etat in donnesEtats){
				$( "#resume_circulaire_" + i + "_" + index).text(etat + " : " + donnesEtats[etat]);
				index++;
			}
		});

		// On récupère les données de l'API - concernant la moyenne - selon l'unite
		$.get( "./api/moyenne_unite/" + i + "/" + unite, function( data ) {
			//$( ".result" ).html( data );
			//alert( "Load was performed." );
			data = data.replace("[", "");
			data = data.replace("]", "");
			var array = data.split(",");
			for (let y=0; y< array.length; y++){
				index = i + $(".continu").length;
				$( "#cellule_circulaire_" + y + "_" + index  ).text(array[y].replace("'", "").replace("'", ""));
			}
		});
		let loi = select_loi.options[select_loi.selectedIndex].value;
		//On met l'image selon la loi et l'unite.
        document.getElementById("image_circulaire_" + i).src = "static/images/plot_circulaire_" + loi + "_" + i + "_" + unite + ".png";
	});
}

/* CAROUSSEL */

// initialisation
function caroussel(divs_c, nom_bouton){
	//On initialise le caroussel si il y a bien des éléments dans la div (e.g autre divs divs_c).
	if (divs_c.length > 0){

		//Sert d'index pour éléments du caroussel
		let current_c = 0;
		//On récupère les boutons 
		let button_c_n = document.getElementById(nom_bouton + "_next");
		let button_c_p = document.getElementById(nom_bouton + "_previous");

		
		// Définir la visibilité sur faux - on initiales les éléments du caroussel
		if(nom_bouton == "circular"){
			for (let i=0; i < divs_c.length; i++){
				//Pour le caroussel avec les plots circulaires, on a besoin de différencier par loi et par unite.
				divs_c[i].style.display='none';
				//On place la première bonne image pour chaque groupe de données (divs)
				let select = document.getElementById('select_loi_' + (i));
				let select_unit = document.getElementById('select_unit_' + (i));
				let loi = select.options[select.selectedIndex].value;
				let unite = select_unit.options[select_unit.selectedIndex].value;
				document.getElementById("image_circulaire_" + (i)).src = "static/images/plot_circulaire_" + loi + "_" + (i) + "_" + unite + ".png";
			}
		}
		else{
			//Pour les caroussels classiques, il n'y a rien à spécifier en plus. Ils sont invisibles fauf le premier
			for (let i=1; i< divs_c.length; i++){
				divs_c[i].style.display='none';
			}	
		}

		// Mettre le premier visible par défaut
		divs_c[0].style.display='flex';

		//Prochain élément du caroussel vers la droite - on switch les divs de invisible à visible et ubversement.
		button_c_n.addEventListener("click", (event) => {
			if (current_c < divs_c.length - 1){
				current_c ++;
				divs_c[current_c - 1].style.display='none';
				divs_c[current_c].style.display='flex';
			}
		});

		//Prochain élément du caroussel vers la gauche.
		button_c_p.addEventListener("click", (event) => {
			if (current_c > 0){
				current_c --;
				divs_c[current_c + 1].style.display='none';
				divs_c[current_c].style.display='flex';
			}
		});

	}
}

// Récupère divs circulaires - plots + resumés circulaires.
let divs_c = $("div[id^='div_circulaire_']");

// Récupère divs discrets - plots + resumés discrets
let divs_d = $("div[id^='div_discrete_']");
// Récupère divs continues - plots + resumés continues
let divs_con = $("div[id^='div_continue_']");

caroussel(divs_c, "circular");
caroussel(divs_d, "discrete");
caroussel(divs_con, "continue");


