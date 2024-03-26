#ifndef IPPL_STRUCTURE_FORMATION_MANAGER_H
#define IPPL_STRUCTURE_FORMATION_MANAGER_H

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
constexpr float G = 4.3009e-09; // Mpc km^2 /s^2 M_Sun

#include "GravityFieldContainer.hpp"
#include "GravityFieldSolver.hpp"
#include "GravityLoadBalancer.hpp"
#include "GravityManager.h"
#include "Manager/BaseManager.h"
#include "GravityParticleContainer.hpp"
#include "Random/Distribution.h"
#include "Random/InverseTransformSampling.h"
#include "Random/NormalDistribution.h"
#include "Random/Randn.h"

using view_type = typename ippl::detail::ViewType<ippl::Vector<double, Dim>, 1>::view_type;
//typedef ippl::ParticleSpatialLayout<double, Dim> PLayout_t;

template <typename T, unsigned Dim>
class StructureFormationManager : public GravityManager<T, Dim> {
public:
    using ParticleContainer_t = ParticleContainer<T, Dim>;
    using FieldContainer_t = FieldContainer<T, Dim>;
    using FieldSolver_t= FieldSolver<T, Dim>;
    using LoadBalancer_t= LoadBalancer<T, Dim>;

    StructureFormationManager(size_type totalP_, int nt_, Vector_t<int, Dim> &nr_,
                       double lbt_, std::string& solver_, std::string& stepMethod_)
        : GravityManager<T, Dim>(totalP_, nt_, nr_, lbt_, solver_, stepMethod_){}

    ~StructureFormationManager(){}

    void pre_run() override {
        Inform mes("Pre Run");

        if (this->solver_m == "OPEN") {
            throw IpplException("StructureFormation", "Open boundaries solver incompatible with this simulation!");
        }

        // Grid 
        for (unsigned i = 0; i < Dim; i++) {
            this->domain_m[i] = ippl::Index(this->nr_m[i]);
        }


        this->decomp_m.fill(true);
        this->rmin_m  = 0.0;
        this->rmax_m  = 50000.0;//2 * pi / this->kw_m; //50000.0;

        this->hr_m = this->rmax_m / this->nr_m;
        // M = -\int\int f dx dv
        this->M_m = std::reduce(this->rmax_m.begin(), this->rmax_m.end(), 1., std::multiplies<double>());
        mes << "total mass: " << this->M_m << endl;
        this->origin_m = this->rmin_m;

        this->Hubble0 =  73.8; // 73.8 km/sec/Mpc
        this->Omega0 = 1.02;
        this->z_m = 64;
        this->InitialiseTime();

        //this->time_m   = 0.0;
        //this->dt_m     = std::min(.05, 0.5 * *std::min_element(this->hr_m.begin(), this->hr_m.end()));
        this->it_m     = 0;
        

        mes << "Discretization:" << endl
          << "nt " << this->nt_m << ", Np = " << this->totalP_m << ", grid = " << this->nr_m << endl;

        this->isAllPeriodic_m = true;

        this->setFieldContainer( std::make_shared<FieldContainer_t>( this->hr_m, this->rmin_m, this->rmax_m, this->decomp_m, this->domain_m, this->origin_m, this->isAllPeriodic_m) );

        this->setParticleContainer( std::make_shared<ParticleContainer_t>( this->fcontainer_m->getMesh(), this->fcontainer_m->getFL()) );

        this->fcontainer_m->initializeFields(this->solver_m);

        this->setFieldSolver( std::make_shared<FieldSolver_t>( this->solver_m, &this->fcontainer_m->getRho(), &this->fcontainer_m->getF(), &this->fcontainer_m->getPhi()) );

        this->fsolver_m->initSolver();

        this->setLoadBalancer( std::make_shared<LoadBalancer_t>( this->lbt_m, this->fcontainer_m, this->pcontainer_m, this->fsolver_m) );

        readParticles(); // defines particle positions, velocities

        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        IpplTimings::startTimer(DummySolveTimer);
        
        this->fcontainer_m->getRho() = 0.0;
        this->fcontainer_m->getRHS() = 0.0;
        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(DummySolveTimer);
        this->par2grid();

        savePositions();

        static IpplTimings::TimerRef SolveTimer = IpplTimings::getTimer("solve");
        IpplTimings::startTimer(SolveTimer);

        this->fsolver_m->runSolver();

        IpplTimings::stopTimer(SolveTimer);

        this->grid2par();

        this->dump();
        

        mes << "Done";
    }
    
    void readParticles() {
        Inform mes("Reading Particles");

        ifstream file("data/Data.csv");

        // Check if the file is opened successfully
        if (!file.is_open()) {
            cerr << "Error opening IC file!" << endl;
        }
        
        
        // Vector to store data read from the CSV file
        vector<unsigned int> ParticleID;
        vector<vector<double>> ParticlePositions;
        vector<vector<double>> ParticleVelocities;
        double MaxPos;
        double MinPos;

        // Read the file line by line
        string line;
        while (getline(file, line)) {
            stringstream ss(line);

            // Read each comma-separated value into the row vector
            string cell;
            unsigned int j = 0;
            vector<double> PosRow;
            vector<double> VelRow;
            unsigned int i = 0;
            while (j < 7 && getline(ss, cell, ',')) {
                if (j == 0){
                    double indexD = stod(cell);
                    //printf("index %f \n", indexD );
                    unsigned int index = (int)indexD;//static_cast<unsigned int>(std::round(indexD));
                    ParticleID.push_back(index);
                }
                else if (j < 4){
                    double Pos = stod(cell);
                    PosRow.push_back(Pos);
                    // Find Boundaries (x, y, z)
                    if(i > 0){
                        MaxPos = max(Pos, MaxPos);
                        MinPos = min(Pos, MinPos);
                    }
                    else{
                        MaxPos = Pos;
                        MinPos = Pos;
                        ++i;
                    }
                }
                else {
                    double Vel = stod(cell);
                    VelRow.push_back(Vel);
                }
                ++j;
                //R_host(index)[j-1] = stoi(cell);
            }
            ParticlePositions.push_back(PosRow);
            ParticleVelocities.push_back(VelRow);
        }

        // Number of Particles 
        unsigned int NumPart = *max_element(ParticleID.begin(), ParticleID.end());
        if (NumPart != ParticleID.size())
            cerr << "Error: Number of Particles does not match indices" << endl;
        else
            mes << "Number of Particles: " << NumPart << endl;

        // Boundaries of Particle Positions
        mes << "Minimum Position: " << MinPos << endl;
        mes << "Maximum Position: " << MaxPos << endl;

        if (this->totalP_m != NumPart || NumPart != ParticleID.size()){
            cerr << "Error: Simulation number of particles does not match input!" << endl;
            cerr << "Input N = " << NumPart << ", Simulation N = " << this->totalP_m << endl;
        }    
        else 
            mes << "successfully done." << endl;

        size_type nloc = this->totalP_m / ippl::Comm->size();
        this->pcontainer_m->create(nloc);
        //ippl::Comm->rank() == 0       
        this->pcontainer_m->m = this->M_m/this->totalP_m;
        
        auto Rview = this->pcontainer_m->R.getView();
        auto Vview = this->pcontainer_m->V.getView();

        Kokkos::parallel_for(
            "Assign initial R", ippl::getRangePolicy(Rview),
            KOKKOS_LAMBDA(const int i) {
                Rview(i)[0] = ParticlePositions[i][0];
                Rview(i)[1] = ParticlePositions[i][1];
                Rview(i)[2] = ParticlePositions[i][2];
                Vview(i)[0] = ParticleVelocities[i][0];
                Vview(i)[1] = ParticleVelocities[i][1];
                Vview(i)[2] = ParticleVelocities[i][2];
            });

        Kokkos::fence();
        mes << "Assignment of positions and velocities done." << endl;

        /*
        using Playout = ippl::ParticleSpatialLayout<T, Dim>;
        using Base = ippl::ParticleBase<Playout>;
        typename Base::particle_position_type::HostMirror R_host = this->pcontainer_m->R.getHostMirror();
        typename Base::particle_position_type::HostMirror P_host = this->pcontainer_m->P.getHostMirror();
        mes << "typenames done. " << endl;
        for (unsigned long int i = 0; i < nloc; i++) {
            for (unsigned d = 0; d < Dim; d++) {
                mes << "part pos " << i << " " << d << " " << ParticlePositions[i][d] << endl;
                mes << "r host " << R_host(i)[d] << endl;
                R_host(i)[d] = ParticlePositions[i][d];
                P_host(i)[d] = ParticleVelocities[i][d];
            }
        }
        // Copy to device
        Kokkos::deep_copy(this->pcontainer_m->R.getView(), R_host);
        Kokkos::deep_copy(this->pcontainer_m->P.getView(), P_host);
        mes << "assignment done. " << endl;
        */

    }

    void advance() override {
        if (this->stepMethod_m == "LeapFrog") {
            LeapFrogStep();
        }
	else{
            throw IpplException(TestName, "Step method is not set/recognized!");
        }
    }

    void LeapFrogStep(){
        // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration

        static IpplTimings::TimerRef VTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");

        double dt                               = this->dt_m;
        double a                                = this->a_m;
        std::shared_ptr<ParticleContainer_t> pc = this->pcontainer_m;
        std::shared_ptr<FieldContainer_t> fc    = this->fcontainer_m;

        // kick (update V)
        IpplTimings::startTimer(VTimer);
        pc->V = pc->V - dt * (0.5/(a*a) * pc->F + this->Hubble_m * pc->V);
        IpplTimings::stopTimer(VTimer);

        // drift (update R) in comoving distances
        IpplTimings::startTimer(RTimer);
        pc->R = pc->R + dt * pc->V;
        IpplTimings::stopTimer(RTimer);

        // Since the particles have moved spatially update them to correct processors
        IpplTimings::startTimer(updateTimer);
        pc->update();
        IpplTimings::stopTimer(updateTimer);
        

        size_type totalP        = this->totalP_m;
        int it                  = this->it_m;
        bool isFirstRepartition = false;
        if (this->loadbalancer_m->balance(totalP, it + 1)) {
                IpplTimings::startTimer(domainDecomposition);
                auto* mesh = &fc->getRho().get_mesh();
                auto* FL = &fc->getFL();
                this->loadbalancer_m->repartition(FL, mesh, isFirstRepartition);
                IpplTimings::stopTimer(domainDecomposition);
        }
        
        // scatter the mass onto the underlying grid
        this->par2grid();

        // Field solve
        IpplTimings::startTimer(SolveTimer);
        this->fsolver_m->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        // gather F field
        this->grid2par();

        // kick (update V)
        IpplTimings::startTimer(VTimer);
        pc->V = pc->V - dt * (0.5/(a*a) * pc->F + this->Hubble_m * pc->V);
        IpplTimings::stopTimer(VTimer);

    }

    void savePositions() {
        Inform mes("Saving Particles");

        ofstream file("data/Positions.csv");

        // Check if the file is opened successfully
        if (!file.is_open()) {
            cerr << "Error opening saving file!" << endl;
            return;
        }

        auto Rview = this->pcontainer_m->R.getView();
        auto Vview = this->pcontainer_m->V.getView();

        // Write data to the file
        for (unsigned int i = 0; i < Rview.size(); ++i){
            file << i << ",";
            for (unsigned int d = 0; d < Dim; ++d)
                file << Rview(i)[d] << ",";
            for (unsigned int d = 0; d < Dim-1; ++d)
                file << Vview(i)[d] << ",";
            file << Vview(i)[Dim-1] << "\n";
        }

        // Close the file stream
        file.close();
        mes << "done." << endl;

    }


    void dump() override {
        static IpplTimings::TimerRef dumpDataTimer = IpplTimings::getTimer("dumpData");
        IpplTimings::startTimer(dumpDataTimer);
        dumpStructure(this->fcontainer_m->getF().getView());
        IpplTimings::stopTimer(dumpDataTimer);
    }

    template <typename View>
    void dumpStructure(const View& Fview) {
        const int nghostF = this->fcontainer_m->getF().getNghost();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        double localEx2 = 0, localExNorm = 0;
        ippl::parallel_reduce(
            "Ex stats", ippl::getRangePolicy(Fview, nghostF),
            KOKKOS_LAMBDA(const index_array_type& args, double& F2, double& FNorm) {
                // ippl::apply<unsigned> accesses the view at the given indices and obtains a
                // reference; see src/Expression/IpplOperations.h
                double val = ippl::apply(Fview, args)[0];
                double f2  = Kokkos::pow(val, 2);
                F2 += f2;

                double norm = Kokkos::fabs(ippl::apply(Fview, args)[0]);
                if (norm > FNorm) {
                    FNorm = norm;
                }
            },
            Kokkos::Sum<double>(localEx2), Kokkos::Max<double>(localExNorm));

        double globaltemp = 0.0;
        ippl::Comm->reduce(localEx2, globaltemp, 1, std::plus<double>());

        double fieldEnergy =
            std::reduce(this->fcontainer_m->getHr().begin(), this->fcontainer_m->getHr().end(), globaltemp, std::multiplies<double>());

        double ExAmp = 0.0;
        ippl::Comm->reduce(localExNorm, ExAmp, 1, std::greater<double>());

        if (ippl::Comm->rank() == 0) {
            std::stringstream fname;
            fname << "data/FieldStructure_";
            fname << ippl::Comm->size();
            fname << "_manager";
            fname << ".csv";
            Inform csvout(NULL, fname.str().c_str(), Inform::APPEND);
            csvout.precision(16);
            csvout.setf(std::ios::scientific, std::ios::floatfield);
            if ( std::fabs(this->time_m) < 1e-14 ) {
                csvout << "time, Ex_field_energy, Ex_max_norm" << endl;
            }
            csvout << this->time_m << " " << fieldEnergy << " " << ExAmp << endl;
        }
        ippl::Comm->barrier();
    }
};
#endif
