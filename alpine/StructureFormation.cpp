// StructureFormation Test
//   Usage:
//     srun ./StructureFormation
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...-direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT and CG supported)
//     lbthres  = Load balancing threshold i.e., lbthres*100 is the maximum load imbalance
//                percentage which can be tolerated and beyond which
//                particle load balancing occurs. A value of 0.01 is good for many typical
//                simulations.
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./StructureFormation 32 32 32 10000 10 FFT 0.01 LeapFrog --overallocate 2.0 --info 5


constexpr unsigned Dim = 3;
using T = double;
const char* TestName = "StructureFormation";

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Ippl.h"
#include "Utility/IpplTimings.h"
#include "Manager/PicManager.h"
#include "datatypes.h"
#include "StructureFormationManager.h"

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(TestName);
        Inform msg2all(TestName, INFORM_ALL_NODES);

        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);

        // Read input parameters, assign them to the corresponding memebers of manager
        int arg = 1;
        // Number of particles in each dimension
        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++) {
            nr[d] = std::atoi(argv[arg++]);
        }
        // Total number of particles
        size_type totalP = std::atoll(argv[arg++]);
        // Number of time steps
        int nt  = std::atoi(argv[arg++]);
        // Solver method
        std::string solver = argv[arg++];
        //
        double lbt = std::atof(argv[arg++]);
        // step method
        std::string step_method = argv[arg++];

        // Create an instance of a manger for the considered application
        StructureFormationManager<T, Dim> manager(totalP, nt, nr, lbt, solver, step_method);

        // Perform pre-run operations, including creating mesh, particles,...
        manager.pre_run();

        manager.setTime(0.0);

        msg << "Starting iterations ..." << endl;

        manager.run(manager.getNt());

        msg << "End." << endl;

        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
